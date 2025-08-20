"""Functions to parse MIDI files and extract events using accelerated backends."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Iterator, overload

import numpy as np

import numba_midi.cython.midi as midi_acc

# Define structured dtype to have a homogenous representation of MIDI events
event_dtype = np.dtype(
    [
        ("tick", np.int32),  # Tick count
        ("event_type", np.uint8),  # Event type (0-6)
        ("channel", np.uint8),  # MIDI Channel (0-15)
        ("value1", np.int32),  # Event-dependent value
        ("value2", np.int16),  # Event-dependent value
        ("value3", np.uint8),  # Event-dependent value
        ("value4", np.uint8),  # Event-dependent value
    ]
)


@dataclass
class Event:
    """MIDI event representation."""

    tick: int
    event_type: int
    channel: int
    value1: int
    value2: int
    value3: int
    value4: int


class EventType(IntEnum):
    """Enum for MIDI event types."""

    note_on = 0
    note_off = 1
    pitch_bend = 2
    control_change = 3
    program_change = 4
    tempo_change = 5
    channel_aftertouch = 6
    polyphonic_aftertouch = 7
    sysex = 8
    end_of_track = 9
    time_signature_change = 10


class Events:
    """Struct of arrays representation for MIDI events - better performance than array of structs."""

    def __init__(
        self,
        tick: np.ndarray,
        event_type: np.ndarray,
        channel: np.ndarray,
        value1: np.ndarray,
        value2: np.ndarray,
        value3: np.ndarray,
        value4: np.ndarray,
    ) -> None:
        """Initialize Events from either individual arrays."""
        # Ensure all arrays have the same length
        arrays = [tick, event_type, channel, value1, value2, value3, value4]
        sizes = [len(arr) for arr in arrays]
        if not all(s == sizes[0] for s in sizes):
            raise ValueError("All arrays must have the same length")

        self._size = sizes[0]
        self._tick = np.asarray(tick, dtype=np.int32)
        self._event_type = np.asarray(event_type, dtype=np.uint8)
        self._channel = np.asarray(channel, dtype=np.uint8)
        self._value1 = np.asarray(value1, dtype=np.int32)
        self._value2 = np.asarray(value2, dtype=np.int16)
        self._value3 = np.asarray(value3, dtype=np.uint8)
        self._value4 = np.asarray(value4, dtype=np.uint8)

    @classmethod
    def zeros(cls, size: int) -> "Events":
        """Create a new Events with zeros."""
        return cls(
            tick=np.zeros(size, dtype=np.int32),
            event_type=np.zeros(size, dtype=np.uint8),
            channel=np.zeros(size, dtype=np.uint8),
            value1=np.zeros(size, dtype=np.int32),
            value2=np.zeros(size, dtype=np.int16),
            value3=np.zeros(size, dtype=np.uint8),
            value4=np.zeros(size, dtype=np.uint8),
        )

    @classmethod
    def concatenate(cls, arrays: Iterable["Events"]) -> "Events":
        """Concatenate multiple Events."""
        if not arrays:
            raise ValueError("No Events to concatenate")

        arrays_list = list(arrays)
        return cls(
            tick=np.concatenate([arr._tick for arr in arrays_list]),
            event_type=np.concatenate([arr._event_type for arr in arrays_list]),
            channel=np.concatenate([arr._channel for arr in arrays_list]),
            value1=np.concatenate([arr._value1 for arr in arrays_list]),
            value2=np.concatenate([arr._value2 for arr in arrays_list]),
            value3=np.concatenate([arr._value3 for arr in arrays_list]),
            value4=np.concatenate([arr._value4 for arr in arrays_list]),
        )

    @property
    def tick(self) -> np.ndarray:
        return self._tick

    @tick.setter
    def tick(self, value: np.ndarray | int) -> None:
        self._tick[:] = value

    @property
    def event_type(self) -> np.ndarray:
        return self._event_type

    @event_type.setter
    def event_type(self, value: np.ndarray | int) -> None:
        self._event_type[:] = value

    @property
    def channel(self) -> np.ndarray:
        return self._channel

    @channel.setter
    def channel(self, value: np.ndarray | int) -> None:
        self._channel[:] = value

    @property
    def value1(self) -> np.ndarray:
        return self._value1

    @value1.setter
    def value1(self, value: np.ndarray | int) -> None:
        self._value1[:] = value

    @property
    def value2(self) -> np.ndarray:
        return self._value2

    @value2.setter
    def value2(self, value: np.ndarray | int) -> None:
        self._value2[:] = value

    @property
    def value3(self) -> np.ndarray:
        return self._value3

    @value3.setter
    def value3(self, value: np.ndarray | int) -> None:
        self._value3[:] = value

    @property
    def value4(self) -> np.ndarray:
        return self._value4

    @value4.setter
    def value4(self, value: np.ndarray | int) -> None:
        self._value4[:] = value

    @overload
    def __getitem__(self, index: int) -> Event:
        pass

    @overload
    def __getitem__(self, index: slice) -> "Events":
        pass

    @overload
    def __getitem__(self, index: np.ndarray) -> "Events":
        pass

    def __getitem__(self, index: int | slice | np.ndarray) -> "Events | Event":
        """Get item(s) from the Events."""
        if isinstance(index, int):
            if index < 0 or index >= self._size:
                raise IndexError("Index out of bounds")
            return Event(
                tick=int(self._tick[index]),
                event_type=int(self._event_type[index]),
                channel=int(self._channel[index]),
                value1=int(self._value1[index]),
                value2=int(self._value2[index]),
                value3=int(self._value3[index]),
                value4=int(self._value4[index]),
            )
        # For slices or boolean arrays, return new Events with sliced arrays
        return Events(
            tick=self._tick[index],
            event_type=self._event_type[index],
            channel=self._channel[index],
            value1=self._value1[index],
            value2=self._value2[index],
            value3=self._value3[index],
            value4=self._value4[index],
        )

    def __setitem__(self, index: int | slice | np.ndarray, value: "Events") -> None:
        self._tick[index] = value._tick
        self._event_type[index] = value._event_type
        self._channel[index] = value._channel
        self._value1[index] = value._value1
        self._value2[index] = value._value2
        self._value3[index] = value._value3
        self._value4[index] = value._value4

    def __len__(self) -> int:
        return self._size

    def as_array(self) -> np.ndarray:
        """Convert struct of arrays back to array of structs for compatibility."""
        result = np.zeros(self._size, dtype=event_dtype)
        result["tick"] = self._tick
        result["event_type"] = self._event_type
        result["channel"] = self._channel
        result["value1"] = self._value1
        result["value2"] = self._value2
        result["value3"] = self._value3
        result["value4"] = self._value4
        return result

    @classmethod
    def from_array(cls, data: np.ndarray) -> "Events":
        return cls(
            tick=data["tick"],
            event_type=data["event_type"],
            channel=data["channel"],
            value1=data["value1"],
            value2=data["value2"],
            value3=data["value3"],
            value4=data["value4"],
        )

    @property
    def size(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"Events(size={self.size})"

    def __iter__(self) -> Iterator[Event]:
        """Iterate over the events."""
        for i in range(self._size):
            yield Event(
                tick=int(self._tick[i]),
                event_type=int(self._event_type[i]),
                channel=int(self._channel[i]),
                value1=int(self._value1[i]),
                value2=int(self._value2[i]),
                value3=int(self._value3[i]),
                value4=int(self._value4[i]),
            )


@dataclass
class MidiTrack:
    """MIDI track representation."""

    name: str
    events: Events  # 1D structured numpy array with event_dtype elements
    lyrics: list[tuple[int, str]] | None  # List of tuples (tick, lyric)

    def __post_init__(self) -> None:
        assert isinstance(self.events, Events), "Events must be a Events"
        assert isinstance(self.name, str), "Track name must be a string"


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


def unpack_uint16_triplet(data: bytes) -> tuple[int, int, int]:
    """Unpacks three 2-byte unsigned integers (big-endian)."""
    return (data[0] << 8) | data[1], (data[2] << 8) | data[3], (data[4] << 8) | data[5]


def load_midi_bytes(data: bytes) -> Midi:
    """Loads MIDI data from a byte array."""
    # Parse header
    if data[:4] != b"MThd":
        raise ValueError("Invalid MIDI file header")

    format_type, num_tracks, ticks_per_quarter = unpack_uint16_triplet(data[8:14])
    # assert format_type == 0, "format_type=0 only supported"
    offset = 14  # Header size is fixed at 14 bytes

    tracks = []
    tracks_names = []
    tracks_lyrics = []
    tracks_events = []

    for _ in range(num_tracks):
        if np.any(data[offset : offset + 4] != b"MTrk"):
            raise ValueError("Invalid track chunk")
        (
            offset,
            midi_events_arrays,
            track_name,
            lyrics,
        ) = midi_acc.parse_midi_track_soa_fast(data, offset)

        tracks_names.append(text_decode(track_name))
        lyrics = [(tick, text_decode(lyric)) for tick, lyric in lyrics] if lyrics else None
        tracks_lyrics.append(lyrics)

        # Create Events directly from the struct of arrays
        tick_array, event_type_array, channel_array, value1_array, value2_array, value3_array, value4_array = (
            midi_events_arrays
        )
        events = Events(
            tick=tick_array,
            event_type=event_type_array,
            channel=channel_array,
            value1=value1_array,
            value2=value2_array,
            value3=value3_array,
            value4=value4_array,
        )
        tracks_events.append(events)

    # modify the tracks to have the same time signature
    for name, lyrics, events in zip(tracks_names, tracks_lyrics, tracks_events, strict=False):
        track = MidiTrack(
            name=name,
            lyrics=lyrics,
            events=events,
        )
        # assert len(midi_events_np)>0, "Track must have at least one event"
        tracks.append(track)

    return Midi(
        tracks=tracks,
        ticks_per_quarter=ticks_per_quarter,
    )


def sort_midi_events(midi_events: Events) -> Events:
    """Sorts MIDI events."""
    order = np.lexsort((midi_events.channel, midi_events.event_type, midi_events.tick))
    sorted_events = midi_events[order]
    return sorted_events


def _encode_midi_track(track: MidiTrack) -> bytes:
    """Encode MIDI track using struct of arrays for better performance."""
    data = midi_acc.encode_midi_track_soa_fast(
        track.name.encode("utf-8"),  # Pre-encode the name to bytes
        track.events.tick,
        track.events.event_type,
        track.events.channel,
        track.events.value1,
        track.events.value2,
        track.events.value3,
        track.events.value4,
    )
    return b"MTrk" + len(data).to_bytes(4, "big") + data.tobytes()


def _encode_midi_track_legacy(track: MidiTrack) -> bytes:
    """Legacy encode function using structured arrays for compatibility."""
    data = midi_acc.encode_midi_track(
        track.name.encode("utf-8"),  # Pre-encode the name to bytes
        track.events.as_array(),
    )
    return b"MTrk" + len(data).to_bytes(4, "big") + data.tobytes()

    return b"MTrk" + len(data).to_bytes(4, "big") + data.tobytes()


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
    for track1, track2 in zip(midi1.tracks, midi2.tracks, strict=False):
        sorted_events1 = sort_midi_events(track1.events)
        sorted_events2 = sort_midi_events(track2.events)
        assert track1.name == track2.name
        assert np.all(sorted_events1.as_array() == sorted_events2.as_array())


def get_event_times(midi_events: np.ndarray, tempo_events: np.ndarray, ticks_per_quarter: int) -> np.ndarray:
    return midi_acc.get_event_times(midi_events, tempo_events, ticks_per_quarter)


def get_event_times_soa(events: Events, tempo_events: Events, ticks_per_quarter: int) -> np.ndarray:
    """Get event times using struct of arrays for better performance."""
    return midi_acc.get_event_times_soa_fast(
        events.tick, events.event_type, tempo_events.tick, tempo_events.value1, ticks_per_quarter
    )
