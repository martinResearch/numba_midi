from dataclasses import dataclass

from numba import njit
from numba.typed import List
import numpy as np

from dataclasses import dataclass

import numpy as np


# Define structured dtype to have a homogenous representation of MIDI events
event_dtype = np.dtype(
    [
        ("delta_time", np.uint32),  # Time difference in ticks
        ("event_type", np.uint8),  # Event type (0-6)
        ("channel", np.uint8),  # MIDI Channel (0-15)
        ("value1", np.int32),  # Event-dependent value
        ("value2", np.int16),  # Event-dependent value
    ]
)

# Define event types and their corresponding value1 and value2 meanings
MIDI_EVENT_TYPES = {
    0: {"name": "Note On", "value1": "Pitch (0-127)", "value2": "Velocity (0-127)"},
    1: {"name": "Note Off", "value1": "Pitch (0-127)", "value2": "Ignored (0)"},
    2: {
        "name": "Pitch Bend",
        "value1": "Bend amount (-8192 to 8191)",
        "value2": "Ignored (0)",
    },
    3: {
        "name": "Control Change",
        "value1": "Control Number (0-127)",
        "value2": "Control Value (0-127)",
    },
    4: {
        "name": "Program Change",
        "value1": "Program Number (0-127)",
        "value2": "Ignored (0)",
    },
    5: {
        "name": "Tempo Change",
        "value1": "Tempo (microseconds per quarter note)",
        "value2": "Ignored (0)",
    },
}


@dataclass
class MidiTrack:
    """MIDI track representation."""

    name: str
    events: np.ndarray  # 1D structured numpy array with event_dtype elements
    ticks_per_quarter: int
    numerator: int
    denominator: int
    clocks_per_click: int
    notated_32nd_notes_per_beat: int

    def __post_init__(self):
        assert self.events.dtype == event_dtype, (
            "Events must be a structured numpy array with event_dtype elements"
        )


@dataclass
class Midi:
    """MIDI score representation."""

    tracks: list[MidiTrack]


@njit(cache=True, boundscheck=False)
def get_even_ticks_and_times(
    midi_events: np.ndarray, ticks_per_quarter: int
) -> np.ndarray:
    """Get the time of each event in ticks and seconds."""
    tick = 0
    time = 0
    second_per_tick = 0.0
    events_ticks = np.zeros((len(midi_events)), dtype=np.float32)
    events_times = np.zeros((len(midi_events)), dtype=np.float32)

    for i in range(len(midi_events)):
        delta_time = midi_events[i]["delta_time"]
        event_type = midi_events[i]["event_type"]

        time += delta_time * second_per_tick
        tick += delta_time
        events_times[i] = time
        events_ticks[i] = tick
        if event_type == 5:
            # tempo change event
            current_tempo = midi_events[i]["value1"]
            second_per_tick = current_tempo / ticks_per_quarter / 1_000_000

    return events_ticks, events_times


@njit(cache=True, boundscheck=False)
def read_var_length(data, offset):
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
def unpack_uint32(data):
    """Unpacks a 4-byte unsigned integer (big-endian)."""
    return (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3]


@njit(cache=True, boundscheck=False)
def unpack_uint8_pair(data):
    """Unpacks two 1-byte unsigned integers."""
    return data[0], data[1]


@njit(cache=True, boundscheck=False)
def unpack_uint16_triplet(data):
    """Unpacks three 2-byte unsigned integers (big-endian)."""
    return (data[0] << 8) | data[1], (data[2] << 8) | data[3], (data[4] << 8) | data[5]


@njit(cache=True, boundscheck=False)
def _parse_midi_track(data, offset, ticks_per_quarter):
    """Parses a MIDI track and accumulates time efficiently with Numba."""
    if unpack_uint32(data[offset : offset + 4]) != unpack_uint32(b"MTrk"):
        raise ValueError("Invalid track chunk")

    track_length = unpack_uint32(data[offset + 4 : offset + 8])
    offset += 8
    track_end = offset + track_length
    midi_events = List()

    while offset < track_end:
        delta_ticks, offset = read_var_length(data, offset)
        status_byte = data[offset]
        offset += 1
        if status_byte == 0xFF:  # Meta event
            meta_type = data[offset]
            offset += 1
            meta_length, offset = read_var_length(data, offset)
            meta_data = data[offset : offset + meta_length]
            offset += meta_length

            if meta_type == 0x51:  # Set Tempo event
                current_tempo = (
                    (meta_data[0] << 16) | (meta_data[1] << 8) | meta_data[2]
                )
                midi_events.append((delta_ticks, 5, 0, current_tempo, 0))

            # time signature
            elif meta_type == 0x58:
                assert meta_length == 4, "Time signature meta event has wrong length"
                (
                    numerator,
                    denominator,
                    clocks_per_click,
                    notated_32nd_notes_per_beat,
                ) = meta_data
            # track name
            elif meta_type == 0x03:
                track_name = meta_data

        elif status_byte == 0xF0:  # SysEx event
            sysex_length, offset = read_var_length(data, offset)
            offset += sysex_length

        elif status_byte in (0xF1, 0xF3):  # 1-byte messages
            offset += 1

        elif status_byte == 0xF2:  # 2-byte message (Song Position Pointer)
            offset += 2

        elif 0x80 <= status_byte <= 0xEF:  # MIDI channel messages
            channel = np.uint8(status_byte & 0x0F)
            message_type = (status_byte & 0xF0) >> 4

            if message_type == 0x9:  # Note On
                pitch, velocity = unpack_uint8_pair(data[offset : offset + 2])
                offset += 2
                midi_events.append((delta_ticks, 0, channel, pitch, velocity))

            elif message_type == 0x8:  # Note Off
                pitch, velocity = unpack_uint8_pair(data[offset : offset + 2])
                offset += 2
                midi_events.append((delta_ticks, 1, channel, pitch, velocity))
            elif message_type == 0xB:  # Control Change
                number, value = unpack_uint8_pair(data[offset : offset + 2])
                offset += 2
                midi_events.append((delta_ticks, 3, channel, number, value))

            elif message_type == 0xC:  # program change
                midi_events.append((delta_ticks, 4, channel, data[offset], 0))
                offset += 1

            elif message_type == 0xE:
                midi_events.append((delta_ticks, 2, channel, data[offset], 0))
                offset += 2
            else:
                offset += 1
        else:
            raise ValueError(f"Invalid status byte: {status_byte}")

    midi_events_np = np.zeros(len(midi_events), dtype=event_dtype)
    for i, event in enumerate(midi_events):
        midi_events_np[i]["delta_time"] = event[0]
        midi_events_np[i]["event_type"] = event[1]
        midi_events_np[i]["channel"] = event[2]
        midi_events_np[i]["value1"] = event[3]
        midi_events_np[i]["value2"] = event[4]

    return (
        offset,
        midi_events_np,
        track_name,
        ticks_per_quarter,
        numerator,
        denominator,
        clocks_per_click,
        notated_32nd_notes_per_beat,
    )


def load_midi_raw(file_path: str) -> Midi:
    """Loads a MIDI file."""
    with open(file_path, "rb") as file:
        data = file.read()
    return load_midi_data(data)


def load_midi_data(data: bytes) -> Midi:
    """Loads MIDI data from a byte array."""
    # Parse header
    if data[:4] != b"MThd":
        raise ValueError("Invalid MIDI file header")

    _, num_tracks, ticks_per_quarter = unpack_uint16_triplet(data[8:14])

    offset = 14  # Header size is fixed at 14 bytes

    tracks = []
    for _ in range(num_tracks):
        if np.any(data[offset : offset + 4] != b"MTrk"):
            raise ValueError("Invalid track chunk")
        (
            offset,
            midi_events_np,
            track_name,
            ticks_per_quarter,
            numerator,
            denominator,
            clocks_per_click,
            notated_32nd_notes_per_beat,
        ) = _parse_midi_track(data, offset, ticks_per_quarter)
        track = MidiTrack(
            name=track_name.decode("utf-8"),
            events=midi_events_np,
            ticks_per_quarter=ticks_per_quarter,
            numerator=numerator,
            denominator=denominator,
            clocks_per_click=clocks_per_click,
            notated_32nd_notes_per_beat=notated_32nd_notes_per_beat,
        )

        tracks.append(track)

    return Midi(tracks=tracks)
