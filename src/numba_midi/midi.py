"""Functions to parse MIDI files and extract events using Numba for performance."""

from dataclasses import dataclass

from numba.core.decorators import njit
from numba.typed import List
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
    numerator: int
    denominator: int
    clocks_per_click: int
    notated_32nd_notes_per_beat: int

    def __post_init__(self) -> None:
        assert self.events.dtype == event_dtype, "Events must be a structured numpy array with event_dtype elements"
        assert isinstance(self.name, str), "Track name must be a string"
        assert isinstance(self.numerator, int), "Numerator must be an integer"
        assert self.numerator > 0, "Numerator must be positive"
        assert isinstance(self.denominator, int), "Denominator must be an integer"
        assert self.denominator > 0, "Denominator must be positive"
        assert isinstance(self.clocks_per_click, int), "Clocks per click must be an integer"
        assert self.clocks_per_click > 0, "Clocks per click must be positive"
        assert isinstance(self.notated_32nd_notes_per_beat, int), "Notated 32nd notes per beat must be an integer"
        assert self.notated_32nd_notes_per_beat > 0, "Notated 32nd notes per beat must be positive"
        assert self.events.ndim == 1, "Events must be a 1D array"


@dataclass
class Midi:
    """MIDI score representation."""

    tracks: list[MidiTrack]
    ticks_per_quarter: int

    def __post_init__(self) -> None:
        assert isinstance(self.tracks, list), "Tracks must be a list of MidiTrack objects"
        for track in self.tracks:
            assert isinstance(track, MidiTrack), "Each track must be a MidiTrack object"
        assert isinstance(self.ticks_per_quarter, int), "ticks_per_quarter must be an integer"
        assert self.ticks_per_quarter > 0, "ticks_per_quarter must be positive"


@njit(cache=True, boundscheck=False)
def get_event_ticks(midi_events: np.ndarray) -> np.ndarray:
    """Get the time of each event in ticks and seconds."""
    tick = np.uint32(0)
    events_ticks = np.zeros((len(midi_events)), dtype=np.uint32)

    for i in range(len(midi_events)):
        delta_time = midi_events[i]["delta_time"]
        tick += delta_time
        events_ticks[i] = tick

    return events_ticks


@njit(cache=True, boundscheck=False)
def get_event_ticks_and_times(midi_events: np.ndarray, ticks_per_quarter: int) -> tuple[np.ndarray, np.ndarray]:
    """Get the time of each event in ticks and seconds."""
    tick = np.uint32(0)
    time = 0.0
    second_per_tick = 0.0
    events_ticks = np.zeros((len(midi_events)), dtype=np.uint32)
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
            microseconds_per_quarter_note = float(midi_events[i]["value1"])
            second_per_tick = microseconds_per_quarter_note / ticks_per_quarter / 1_000_000

    return events_ticks, events_times


@njit(cache=True, boundscheck=False)
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
def _parse_midi_track(data: bytes, offset: int) -> tuple:
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
                current_tempo = (meta_data[0] << 16) | (meta_data[1] << 8) | meta_data[2]
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
            elif meta_type == 0x01:
                # Text event
                pass

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
        numerator,
        denominator,
        clocks_per_click,
        notated_32nd_notes_per_beat,
    )


def load_midi_score(file_path: str) -> Midi:
    """Loads a MIDI file."""
    with open(file_path, "rb") as file:
        data = file.read()
    return load_midi_bytes(data)


def load_midi_bytes(data: bytes) -> Midi:
    """Loads MIDI data from a byte array."""
    # Parse header
    if data[:4] != b"MThd":
        raise ValueError("Invalid MIDI file header")

    format_type, num_tracks, ticks_per_quarter = unpack_uint16_triplet(data[8:14])
    assert format_type == 0, "formt_type=0 only supported"
    offset = 14  # Header size is fixed at 14 bytes

    tracks = []
    for _ in range(num_tracks):
        if np.any(data[offset : offset + 4] != b"MTrk"):
            raise ValueError("Invalid track chunk")
        (
            offset,
            midi_events_np,
            track_name,
            numerator,
            denominator,
            clocks_per_click,
            notated_32nd_notes_per_beat,
        ) = _parse_midi_track(data, offset)
        track = MidiTrack(
            name=track_name.decode("utf-8"),
            events=midi_events_np,
            numerator=numerator,
            denominator=denominator,
            clocks_per_click=clocks_per_click,
            notated_32nd_notes_per_beat=notated_32nd_notes_per_beat,
        )

        tracks.append(track)

    return Midi(tracks=tracks, ticks_per_quarter=ticks_per_quarter)


def sort_midi_events(midi_events: np.ndarray) -> np.ndarray:
    """Sorts MIDI events."""
    ticks = get_event_ticks(midi_events)
    order = np.lexsort((midi_events["channel"], midi_events["event_type"], ticks))
    sorted_events = midi_events[order]
    sorted_ticks = ticks[order]
    delta_time = np.diff(sorted_ticks, prepend=0)
    sorted_events["delta_time"] = delta_time
    return sorted_events


def encode_delta_time(delta_time: int) -> bytes:
    """Encodes delta time as a variable-length quantity."""
    if delta_time == 0:
        return b"\x00"
    result = bytearray()
    while delta_time > 0:
        byte = delta_time & 0x7F
        delta_time >>= 7
        if result:
            byte |= 0x80
        result.append(byte)
    return bytes(result[::-1])


def _encode_midi_track(track: MidiTrack) -> bytes:
    """Encodes a MIDI track to bytes."""
    data = b""

    # add track name
    data += encode_delta_time(0)
    track_name = track.name.encode("utf-8")
    data += bytes([0xFF, 0x03]) + bytes([len(track_name)]) + track_name

    # add time signature
    data += encode_delta_time(0)
    data += bytes([0xFF, 0x58, 4]) + bytes(
        [track.numerator, track.denominator, track.clocks_per_click, track.notated_32nd_notes_per_beat]
    )
    # # add tempo change
    # data += bytes([0xFF, 0x51, 3]) + bytes([0x07, 0xA1, 0x20])  # 120 BPM

    for event in track.events:
        delta_time = event["delta_time"]
        event_type = event["event_type"]
        channel = event["channel"]
        value1 = event["value1"]
        value2 = event["value2"]

        data += encode_delta_time(delta_time)  # delta time for track name

        if event_type == 0:
            # Note On
            data += bytes([0x90 | channel]) + bytes([value1, value2])
        elif event_type == 1:
            # Note Off
            data += bytes([0x80 | channel]) + bytes([value1, value2])
        elif event_type == 2:
            # Pitch Bend
            data += bytes([0xE0 | channel]) + bytes([value1, 0])
        elif event_type == 3:
            # Control Change
            data += bytes([0xB0 | channel]) + bytes([value1, value2])
        elif event_type == 4:
            # Program Change
            data += bytes([0xC0 | channel]) + bytes([value1])
        elif event_type == 5:
            # Tempo Change
            data += bytes([0xFF, 0x51, 3]) + bytes([value1 >> 16, (value1 >> 8) & 0xFF, value1 & 0xFF])
        elif event_type == 6:
            # Meta event (e.g., track name)
            data += bytes([0xFF, 0x03]) + bytes([len(value1)]) + value1
        else:
            raise ValueError(f"Invalid event type: {event_type}")
    return b"MTrk" + len(data).to_bytes(4, "big") + data


def save_midi_data(midi: Midi) -> bytes:
    """Saves MIDI data to a byte array."""
    midi_bytes = b"MThd"

    # encode num_tracks and ticks_per_quarter
    num_tracks = 1
    ticks_per_quarter = midi.ticks_per_quarter
    midi_bytes += b"\x00\x00\x00\x06\x00\x00" + num_tracks.to_bytes(2, "big") + ticks_per_quarter.to_bytes(2, "big")

    for track in midi.tracks:
        midi_bytes += _encode_midi_track(track)
    return midi_bytes


def assert_midi_equal(midi1: Midi, midi2: Midi) -> None:
    """Check if two midi files are equal."""
    assert midi1.ticks_per_quarter == midi2.ticks_per_quarter
    assert len(midi1.tracks) == len(midi2.tracks)
    for track1, track2 in zip(midi1.tracks, midi2.tracks):
        sorted_events1 = sort_midi_events(track1.events)
        sorted_events2 = sort_midi_events(track2.events)
        assert track1.name == track2.name
        assert track1.numerator == track2.numerator
        assert track1.denominator == track2.denominator
        assert track1.clocks_per_click == track2.clocks_per_click
        assert track1.notated_32nd_notes_per_beat == track2.notated_32nd_notes_per_beat
        assert np.all(sorted_events1 == sorted_events2)
