"""Functions to parse MIDI files and extract events using Numba for performance."""

from numba.core.decorators import njit
from numba.typed import List
import numpy as np

# Define structured dtype to have a homogenous representation of MIDI events
event_dtype = np.dtype(
    [
        ("tick", np.uint32),  # Tick count
        ("event_type", np.uint8),  # Event type (0-6)
        ("channel", np.uint8),  # MIDI Channel (0-15)
        ("value1", np.int32),  # Event-dependent value
        ("value2", np.int16),  # Event-dependent value
        ("value3", np.int8),  # Event-dependent value
        ("value4", np.int8),  # Event-dependent value
    ]
)


@njit(cache=True, boundscheck=False)
def get_event_times_jit(midi_events: np.ndarray, tempo_events: np.ndarray, ticks_per_quarter: int) -> np.ndarray:
    """Get the time of each event in ticks and seconds."""
    tick = np.uint32(0)
    time = 0.0
    quarter_notes_per_minute_init = 120.0  # Default tempo for the first event
    second_per_tick = quarter_notes_per_minute_init / ticks_per_quarter / 60.0
    events_times = np.zeros((len(midi_events)), dtype=np.float32)

    ref_tick = 0
    ref_time = 0.0
    last_tempo_event = -1

    for i in range(len(midi_events)):
        delta_tick = midi_events[i]["tick"] - tick
        tick += delta_tick
        while last_tempo_event + 1 < len(tempo_events) and tick >= tempo_events[last_tempo_event + 1]["tick"]:
            # tempo change event
            last_tempo_event += 1
            tempo_event = tempo_events[last_tempo_event]
            ref_time = ref_time + (tempo_event["tick"] - ref_tick) * second_per_tick
            ref_tick = tempo_event["tick"]
            microseconds_per_quarter_note = float(tempo_events[last_tempo_event]["value1"])
            second_per_tick = microseconds_per_quarter_note / ticks_per_quarter / 1_000_000

        time = ref_time + (tick - ref_tick) * second_per_tick
        events_times[i] = time

    return events_times


@njit(cache=True, boundscheck=True)
def read_var_length(data: bytes, offset: int) -> tuple[int, int]:
    """Reads a variable-length quantity from the MIDI file."""
    value = 0
    while True:
        byte = data[offset]
        value = (value << 7) | (byte & 0x7F)
        offset += 1
        if byte & 0x80 == 0:
            break
    return value, offset


@njit(cache=True, boundscheck=False)
def unpack_uint32(data: bytes) -> int:
    """Unpacks a 4-byte unsigned integer (big-endian)."""
    return (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3]


@njit(cache=True, boundscheck=False)
def unpack_uint8_pair(data: bytes) -> tuple[int, int]:
    """Unpacks two 1-byte unsigned integers."""
    return data[0], data[1]


@njit(cache=True, boundscheck=False)
def unpack_uint16_triplet(data: bytes) -> tuple[int, int, int]:
    """Unpacks three 2-byte unsigned integers (big-endian)."""
    return (data[0] << 8) | data[1], (data[2] << 8) | data[3], (data[4] << 8) | data[5]


@njit(cache=True, boundscheck=False)
def decode_pitch_bend(data: bytes) -> np.int32:
    assert 0 <= data[0] <= 127
    assert 0 <= data[1] <= 127
    unsigned = (data[1] << 7) | data[0]
    return np.int32(unsigned - 8192)


@njit(cache=True, boundscheck=False)
def encode_pitchbend(value: int) -> tuple[int, int]:
    """Encodes a pitch bend value to two bytes."""
    assert -8192 <= value <= 8191, "Pitch bend value out of range"
    unsigned = value + 8192
    byte1 = unsigned & 0x7F
    byte2 = (unsigned >> 7) & 0x7F
    return byte1, byte2


@njit(cache=True, boundscheck=True)
def parse_midi_track(data: bytes, offset: int) -> tuple:
    """Parses a MIDI track and accumulates time efficiently with Numba."""
    if unpack_uint32(data[offset : offset + 4]) != unpack_uint32(b"MTrk"):
        raise ValueError("Invalid track chunk")

    track_length = unpack_uint32(data[offset + 4 : offset + 8])
    offset += 8
    assert track_length > 0, "Track length must be positive"
    track_end = offset + track_length
    assert track_end <= len(data), "Track length too large."

    midi_events: list[tuple[np.uint32, np.uint8, np.uint8, np.int32, np.int16, np.uint8, np.uint8]] = List()
    track_name = b""
    tick = np.uint32(0)

    lyrics = []

    while offset < track_end:
        _delta_ticks, offset = read_var_length(data, offset)
        tick += _delta_ticks
        status_byte = data[offset]
        offset += 1
        if status_byte == 0xFF:  # Meta event
            meta_type = data[offset]
            offset += 1
            meta_length, offset = read_var_length(data, offset)
            meta_data = data[offset : offset + meta_length]
            offset += meta_length

            if meta_type == 0x51:  # Set Tempo event
                current_tempo = (meta_data[0] << 16) | (meta_data[1] << 8) | meta_data[2]
                midi_events.append(
                    (tick, np.uint8(5), np.uint8(0), np.int32(current_tempo), np.int16(0), np.uint8(0), np.uint8(0))
                )

            # time signature
            elif meta_type == 0x58:
                assert meta_length == 4, "Time signature meta event has wrong length"
                # assert numerator == 0 and denominator == 0, "Multiple time signatures not supported"
                (
                    numerator,
                    denominator_power_of_2,
                    clocks_per_click,
                    notated_32nd_notes_per_beat,
                ) = meta_data

                midi_events.append(
                    (
                        tick,
                        np.uint8(10),
                        np.uint8(0),
                        np.int32(numerator),
                        np.int16(denominator_power_of_2),
                        np.uint8(clocks_per_click),
                        np.uint8(notated_32nd_notes_per_beat),
                    )
                )

            # track name
            elif meta_type == 0x59:
                # sharps = meta_data[0]
                # minor = meta_data[1]
                pass
            elif meta_type == 0x03:
                track_name = meta_data
            elif meta_type == 0x01:
                # Text event
                text = meta_data
                if not text.startswith(b"@") and not text.startswith(b"%") and tick > 0:
                    lyrics.append((tick, text))
                pass
            elif meta_type == 0x04:
                # Lyric event
                pass
            elif meta_type == 0x2F:  # End of track
                midi_events.append((tick, np.uint8(9), np.uint8(0), np.int32(0), np.int16(0), np.uint8(0), np.uint8(0)))

        elif status_byte == 0xF0:  # SysEx event
            # System Exclusive (aka SysEx) messages are used to send device specific data.
            sysex_length, offset = read_var_length(data, offset)
            offset += sysex_length

        elif status_byte in (0xF1, 0xF3):  # 1-byte messages
            offset += 1

        elif status_byte == 0xF2:  # 2-byte message (Song Position Pointer)
            offset += 2

        elif status_byte == 0xF8:  # Clock
            offset += 1
        elif status_byte == 0xFA:  # Start
            offset += 1

        elif status_byte == 0xFC:  # Continue
            offset += 1

        elif status_byte <= 0xEF:  # MIDI channel messages
            if status_byte >= 0x80:
                channel = np.uint8(status_byte & 0x0F)
                message_type = (status_byte & 0xF0) >> 4
            else:
                # running status: use the last event type and channel
                offset -= 1

            if message_type == 0x9:  # Note On
                pitch, velocity = unpack_uint8_pair(data[offset : offset + 2])
                offset += 2
                midi_events.append((tick, np.uint8(0), channel, pitch, velocity, np.uint8(0), np.uint8(0)))

            elif message_type == 0x8:  # Note Off
                pitch, velocity = unpack_uint8_pair(data[offset : offset + 2])
                offset += 2
                midi_events.append((tick, np.uint8(1), channel, pitch, velocity, np.uint8(0), np.uint8(0)))

            elif message_type == 0xB:  # Control Change
                number, value = unpack_uint8_pair(data[offset : offset + 2])
                offset += 2
                midi_events.append((tick, np.uint8(3), channel, number, value, np.uint8(0), np.uint8(0)))

            elif message_type == 0xC:  # program change
                program = np.int32(data[offset])
                midi_events.append((tick, np.uint8(4), channel, program, np.int16(0), np.uint8(0), np.uint8(0)))
                offset += 1

            elif message_type == 0xE:  # Pitch Bend
                value = decode_pitch_bend(data[offset : offset + 2])
                assert value >= -8192 and value <= 8191, "Pitch bend value out of range"
                midi_events.append((tick, np.uint8(2), channel, np.int32(value), np.int16(0), np.uint8(0), np.uint8(0)))
                offset += 2

            elif message_type == 0xA:  # Polyphonic Aftertouch
                pitch, pressure = unpack_uint8_pair(data[offset : offset + 2])
                offset += 2
                midi_events.append((tick, np.uint8(7), channel, pitch, pressure, np.uint8(0), np.uint8(0)))

            elif message_type == 0xD:  # Channel Aftertouch
                pressure = data[offset]
                offset += 1
                midi_events.append((tick, np.uint8(6), channel, pressure, np.int16(0), np.uint8(0), np.uint8(0)))
            else:
                offset += 1

        else:
            raise ValueError(f"Invalid status byte: {status_byte}")

    # assert len(midi_events) > 0, "Track must have at least one event"
    midi_events_np = np.zeros(len(midi_events), dtype=event_dtype)
    for i, event in enumerate(midi_events):
        midi_events_np[i]["tick"] = event[0]
        midi_events_np[i]["event_type"] = event[1]
        midi_events_np[i]["channel"] = event[2]
        midi_events_np[i]["value1"] = event[3]
        midi_events_np[i]["value2"] = event[4]
        midi_events_np[i]["value3"] = event[5]
        midi_events_np[i]["value4"] = event[6]

    return (
        offset,
        midi_events_np,
        track_name,
        lyrics,
    )


@njit(cache=True, boundscheck=False)
def encode_delta_time(delta_time: int) -> List:
    """Encodes delta time as a variable-length quantity."""
    if delta_time == 0:
        return List([np.uint8(0)])
    result = List.empty_list(np.uint8)
    while delta_time > 0:
        byte = delta_time & 0x7F
        delta_time >>= 7
        if len(result) > 0:
            byte |= 0x80
        result.insert(0, np.uint8(byte))
    return result


@njit(cache=True, boundscheck=False)
def encode_midi_track_numba(
    name: bytes,
    events: np.ndarray,
) -> np.ndarray:
    """Encodes a MIDI track to bytes."""
    data = []

    # Add track name
    data.extend(encode_delta_time(0))
    data.extend([0xFF, 0x03, len(name)])
    data.extend(name)

    tick = np.uint32(0)
    for event in events:
        delta_time = event["tick"] - tick
        tick = event["tick"]
        event_type = event["event_type"]
        channel = event["channel"]
        value1 = event["value1"]
        value2 = event["value2"]
        value3 = event["value3"]
        value4 = event["value4"]

        data.extend(encode_delta_time(delta_time))  # Delta time for the event

        if event_type == 0:
            # Note On
            data.extend([0x90 | channel, value1, value2])
        elif event_type == 1:
            # Note Off
            data.extend([0x80 | channel, value1, value2])
        elif event_type == 2:
            # Pitch Bend
            d = encode_pitchbend(value1)
            data.extend([0xE0 | channel, d[0], d[1]])
        elif event_type == 3:
            # Control Change
            data.extend([0xB0 | channel, value1, value2])
        elif event_type == 4:
            # Program Change
            data.extend([0xC0 | channel, value1])
        elif event_type == 5:
            # Tempo Change
            data.extend([0xFF, 0x51, 3, value1 >> 16, (value1 >> 8) & 0xFF, value1 & 0xFF])
        elif event_type == 6:
            # Channel Aftertouch
            data.extend([0xD0 | channel, value1])
        elif event_type == 9:
            # End of track
            data.extend([0xFF, 0x2F, 0])
        elif event_type == 10:
            # Time Signature Change
            data.extend([0xFF, 0x58, 4, value1, value2, value3, value4])
        else:
            raise ValueError(f"Invalid event type: {event_type}")

    return np.array(data, dtype=np.uint8)
