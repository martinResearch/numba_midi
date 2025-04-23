"""Functions to parse MIDI files and extract events using Numba for performance."""

from dataclasses import dataclass

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
    ]
)


class EventArray:
    """Wrapper for a structured numpy array with event_dtype elements."""

    def __init__(self, data: np.ndarray) -> None:
        if data.dtype != event_dtype:
            raise ValueError("Invalid dtype for ControlArray")
        self._data = data

    @classmethod
    def zeros(cls, size: int) -> "EventArray":
        """Create a new EventArray with zeros."""
        data = np.zeros(size, dtype=event_dtype)
        return cls(data)

    @classmethod
    def concatenate(cls, arrays: list["EventArray"]) -> "EventArray":
        """Concatenate multiple EventArrays."""
        if not arrays:
            raise ValueError("No EventArrays to concatenate")
        data = np.concatenate([arr._data for arr in arrays])
        return cls(data)

    @property
    def tick(self) -> np.ndarray:
        return self._data["tick"]

    @tick.setter
    def tick(self, value: np.ndarray | int) -> None:
        self._data["tick"][:] = value

    @property
    def event_type(self) -> np.ndarray:
        return self._data["event_type"]

    @event_type.setter
    def event_type(self, value: np.ndarray | int) -> None:
        self._data["event_type"][:] = value

    @property
    def channel(self) -> np.ndarray:
        return self._data["channel"]

    @channel.setter
    def channel(self, value: np.ndarray | int) -> None:
        self._data["channel"][:] = value

    @property
    def value1(self) -> np.ndarray:
        return self._data["value1"]

    @value1.setter
    def value1(self, value: np.ndarray | int) -> None:
        self._data["value1"][:] = value

    @property
    def value2(self) -> np.ndarray:
        return self._data["value2"]

    @value2.setter
    def value2(self, value: np.ndarray | int) -> None:
        self._data["value2"][:] = value

    def __getitem__(self, index: int | slice | np.ndarray) -> "EventArray":
        result = self._data[index]
        return EventArray(result)  # Return new wrapper for slices or boolean arrays

    def __setitem__(self, index: int | slice | np.ndarray, value: "EventArray") -> None:
        self._data[index] = value._data

    def __len__(self) -> int:
        return len(self._data)

    def as_array(self) -> np.ndarray:
        return self._data

    @property
    def size(self) -> int:
        return self._data.size


# event types:
# 0: Note On
# 1: Note Off
# 2: Pitch Bend
# 3: Control Change
# 4: Program Change
# 5: Tempo Change
# 6: Channel Aftertouch
# 7: Polyphonic Aftertouch
# 8: SysEx


@dataclass
class MidiTrack:
    """MIDI track representation."""

    name: str
    events: EventArray  # 1D structured numpy array with event_dtype elements
    lyrics: list[tuple[int, str]] | None  # List of tuples (tick, lyric)
    time_signature: tuple[int, int]
    clocks_per_click: int
    notated_32nd_notes_per_beat: int

    def __post_init__(self) -> None:
        assert isinstance(self.events, EventArray), "Events must be a EventArray"
        assert isinstance(self.name, str), "Track name must be a string"
        assert isinstance(self.time_signature[0], int), "Numerator must be an integer"
        assert self.time_signature[0] > 0, "Numerator must be positive"
        assert isinstance(self.time_signature[1], int), "Denominator must be an integer"
        assert self.time_signature[1] > 0, "Denominator must be positive"
        assert isinstance(self.clocks_per_click, int), "Clocks per click must be an integer"
        # assert self.clocks_per_click > 0, "Clocks per click must be positive"
        assert isinstance(self.notated_32nd_notes_per_beat, int), "Notated 32nd notes per beat must be an integer"
        assert self.notated_32nd_notes_per_beat > 0, "Notated 32nd notes per beat must be positive"


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

    def __repr__(self) -> str:
        num_events = sum(len(track.events) for track in self.tracks)
        return f"Midi(num_tracks={len(self.tracks)}, num_events={num_events})"

    @classmethod
    def from_file(cls, file_path: str) -> "Midi":
        """Load a MIDI file."""
        with open(file_path, "rb") as file:
            data = file.read()
        return cls.from_bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Midi":
        return load_midi_bytes(data)

    def to_bytes(self) -> bytes:
        """Convert the MIDI object to bytes."""
        return save_midi_data(self)

    def save(self, file_path: str) -> None:
        """Save the MIDI object to a file."""
        with open(file_path, "wb") as file:
            file.write(self.to_bytes())


@njit(cache=True, boundscheck=False)
def get_event_times(midi_events: np.ndarray, tempo_events: np.ndarray, ticks_per_quarter: int) -> np.ndarray:
    """Get the time of each event in ticks and seconds."""
    tick = np.uint32(0)
    time = 0.0
    second_per_tick = 0.0
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


@njit(cache=True, boundscheck=True)
def _parse_midi_track(data: bytes, offset: int) -> tuple:
    """Parses a MIDI track and accumulates time efficiently with Numba."""
    if unpack_uint32(data[offset : offset + 4]) != unpack_uint32(b"MTrk"):
        raise ValueError("Invalid track chunk")

    track_length = unpack_uint32(data[offset + 4 : offset + 8])
    offset += 8
    assert track_length > 0, "Track length must be positive"
    track_end = offset + track_length
    assert track_end <= len(data), "Track length too large."
    midi_events = List()
    track_name = b""
    tick = np.uint32(0)
    numerator = 4
    denominator = 4
    clocks_per_click = 24
    notated_32nd_notes_per_beat = 8
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
                midi_events.append((tick, 5, 0, current_tempo, 0))

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
                pass

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
                midi_events.append((tick, 0, channel, pitch, velocity))

            elif message_type == 0x8:  # Note Off
                pitch, velocity = unpack_uint8_pair(data[offset : offset + 2])
                offset += 2
                midi_events.append((tick, 1, channel, pitch, velocity))
            elif message_type == 0xB:  # Control Change
                number, value = unpack_uint8_pair(data[offset : offset + 2])
                offset += 2
                midi_events.append((tick, 3, channel, number, value))

            elif message_type == 0xC:  # program change
                midi_events.append((tick, 4, channel, data[offset], 0))
                offset += 1

            elif message_type == 0xE:
                midi_events.append((tick, 2, channel, data[offset], 0))
                offset += 2

            elif message_type == 0xA:  # Polyphonic Aftertouch
                pitch, pressure = unpack_uint8_pair(data[offset : offset + 2])
                offset += 2
                midi_events.append((tick, 7, channel, pitch, pressure))

            elif message_type == 0xD:  # Channel Aftertouch
                pressure = data[offset]
                offset += 1
                midi_events.append((tick, 6, channel, pressure, 0))
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

    return (
        offset,
        midi_events_np,
        track_name,
        numerator,
        denominator,
        clocks_per_click,
        notated_32nd_notes_per_beat,
        lyrics,
    )


def load_midi_score(file_path: str) -> Midi:
    """Loads a MIDI file."""
    with open(file_path, "rb") as file:
        data = file.read()
    return load_midi_bytes(data)


def text_decode(data: bytes) -> str:
    """Decodes a byte array to a string."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def load_midi_bytes(data: bytes) -> Midi:
    """Loads MIDI data from a byte array."""
    # Parse header
    if data[:4] != b"MThd":
        raise ValueError("Invalid MIDI file header")

    format_type, num_tracks, ticks_per_quarter = unpack_uint16_triplet(data[8:14])
    # assert format_type == 0, "format_type=0 only supported"
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
            lyrics,
        ) = _parse_midi_track(data, offset)

        name = text_decode(track_name)
        lyrics = [(tick, text_decode(lyric)) for tick, lyric in lyrics] if lyrics else None
        track = MidiTrack(
            name=name,
            lyrics=lyrics,
            events=EventArray(midi_events_np),
            time_signature=(numerator, denominator),
            clocks_per_click=clocks_per_click,
            notated_32nd_notes_per_beat=notated_32nd_notes_per_beat,
        )
        # assert len(midi_events_np)>0, "Track must have at least one event"
        tracks.append(track)

    return Midi(tracks=tracks, ticks_per_quarter=ticks_per_quarter)


def sort_midi_events(midi_events: EventArray) -> EventArray:
    """Sorts MIDI events."""
    order = np.lexsort((midi_events.channel, midi_events.event_type, midi_events.tick))
    sorted_events = midi_events[order]
    return sorted_events


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


def _encode_midi_track(track: MidiTrack) -> bytes:
    data = _encode_midi_track_numba(
        track.name.encode("utf-8"),  # Pre-encode the name to bytes
        track.time_signature[0],
        track.time_signature[1],
        track.clocks_per_click,
        track.notated_32nd_notes_per_beat,
        track.events._data,
    )
    return b"MTrk" + len(data).to_bytes(4, "big") + data.tobytes()


@njit(cache=True, boundscheck=False)
def _encode_midi_track_numba(
    name: bytes,
    numerator: int,
    denominator: int,
    clocks_per_click: int,
    notated_32nd_notes_per_beat: int,
    events: np.ndarray,
) -> np.ndarray:
    """Encodes a MIDI track to bytes."""
    data = []

    # Add track name
    data.extend(encode_delta_time(0))
    data.extend([0xFF, 0x03, len(name)])
    data.extend(name)

    # Add time signature
    data.extend(encode_delta_time(0))
    data.extend([0xFF, 0x58, 4, numerator, denominator, clocks_per_click, notated_32nd_notes_per_beat])

    tick = np.uint32(0)
    for event in events:
        delta_time = event["tick"] - tick
        tick = event["tick"]
        event_type = event["event_type"]
        channel = event["channel"]
        value1 = event["value1"]
        value2 = event["value2"]

        data.extend(encode_delta_time(delta_time))  # Delta time for the event

        if event_type == 0:
            # Note On
            data.extend([0x90 | channel, value1, value2])
        elif event_type == 1:
            # Note Off
            data.extend([0x80 | channel, value1, value2])
        elif event_type == 2:
            # Pitch Bend
            data.extend([0xE0 | channel, value1, 0])
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
        else:
            raise ValueError(f"Invalid event type: {event_type}")

    return np.array(data, dtype=np.uint8)


def save_midi_data(midi: Midi) -> bytes:
    """Saves MIDI data to a byte array."""
    midi_bytes = b"MThd"

    # encode num_tracks and ticks_per_quarter
    num_tracks = len(midi.tracks)
    ticks_per_quarter = midi.ticks_per_quarter
    midi_bytes += b"\x00\x00\x00\x06\x00\x00" + num_tracks.to_bytes(2, "big") + ticks_per_quarter.to_bytes(2, "big")

    for track in midi.tracks:
        midi_bytes += _encode_midi_track(track)
    return midi_bytes


def save_midi_file(midi: Midi, file_path: str) -> None:
    """Saves MIDI data to a file."""
    midi_bytes = save_midi_data(midi)
    with open(file_path, "wb") as file:
        file.write(midi_bytes)


def assert_midi_equal(midi1: Midi, midi2: Midi) -> None:
    """Check if two midi files are equal."""
    assert midi1.ticks_per_quarter == midi2.ticks_per_quarter
    assert len(midi1.tracks) == len(midi2.tracks)
    for track1, track2 in zip(midi1.tracks, midi2.tracks):
        sorted_events1 = sort_midi_events(track1.events)
        sorted_events2 = sort_midi_events(track2.events)
        assert track1.name == track2.name
        assert track1.time_signature == track2.time_signature
        assert track1.clocks_per_click == track2.clocks_per_click
        assert track1.notated_32nd_notes_per_beat == track2.notated_32nd_notes_per_beat
        assert np.all(sorted_events1._data == sorted_events2._data)
