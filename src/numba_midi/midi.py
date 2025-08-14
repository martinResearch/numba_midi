"""Functions to parse MIDI files and extract events using accelerated backends."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Iterator, overload

import numpy as np

#from numba_midi.numba.midi import encode_midi_track_numba, event_dtype, parse_midi_track, unpack_uint16_triplet, get_event_times_jit
import numba_midi.cython.midi as midi_acc 

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
    """Wrapper for a structured numpy array with event_dtype elements."""

    def __init__(self, data: np.ndarray) -> None:
        if data.dtype != event_dtype:
            raise ValueError("Invalid dtype for Controls")
        self._data = data

    @classmethod
    def zeros(cls, size: int) -> "Events":
        """Create a new Events with zeros."""
        data = np.zeros(size, dtype=event_dtype)
        return cls(data)

    @classmethod
    def concatenate(cls, arrays: Iterable["Events"]) -> "Events":
        """Concatenate multiple Eventss."""
        if not arrays:
            raise ValueError("No Eventss to concatenate")
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

    @property
    def value3(self) -> np.ndarray:
        return self._data["value3"]

    @value3.setter
    def value3(self, value: np.ndarray | int) -> None:
        self._data["value3"][:] = value

    @property
    def value4(self) -> np.ndarray:
        return self._data["value4"]

    @value4.setter
    def value4(self, value: np.ndarray | int) -> None:
        self._data["value4"][:] = value

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
            if index < 0 or index >= len(self._data):
                raise IndexError("Index out of bounds")
            return Event(
                tick=self._data["tick"][index],
                event_type=self._data["event_type"][index],
                channel=self._data["channel"][index],
                value1=self._data["value1"][index],
                value2=self._data["value2"][index],
                value3=self._data["value3"][index],
                value4=self._data["value4"][index],
            )
        result = self._data[index]
        return Events(result)  # Return new wrapper for slices or boolean arrays

    def __setitem__(self, index: int | slice | np.ndarray, value: "Events") -> None:
        self._data[index] = value._data

    def __len__(self) -> int:
        return len(self._data)

    def as_array(self) -> np.ndarray:
        return self._data

    @property
    def size(self) -> int:
        return self._data.size

    def __repr__(self) -> str:
        return f"Events(size={self.size})"

    def __iter__(self) -> Iterator[Event]:
        """Iterate over the events."""
        for event in self._data:
            yield Event(
                tick=event["tick"],
                event_type=event["event_type"],
                channel=event["channel"],
                value1=event["value1"],
                value2=event["value2"],
                value3=event["value3"],
                value4=event["value4"],
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


def load_midi_bytes(data: bytes) -> Midi:
    """Loads MIDI data from a byte array."""
    # Parse header
    if data[:4] != b"MThd":
        raise ValueError("Invalid MIDI file header")

    format_type, num_tracks, ticks_per_quarter = midi_acc.unpack_uint16_triplet(data[8:14])
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
            midi_events_np,
            track_name,
            lyrics,
        ) = midi_acc.parse_midi_track(data, offset)

        tracks_names.append(text_decode(track_name))
        lyrics = [(tick, text_decode(lyric)) for tick, lyric in lyrics] if lyrics else None
        tracks_lyrics.append(lyrics)
        tracks_events.append(Events(midi_events_np))

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
    data = midi_acc.encode_midi_track(
        track.name.encode("utf-8"),  # Pre-encode the name to bytes
        track.events._data,
    )
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
        assert np.all(sorted_events1._data == sorted_events2._data)



def get_event_times(midi_events: np.ndarray, tempo_events: np.ndarray, ticks_per_quarter: int) -> np.ndarray:
    return midi_acc.get_event_times(midi_events, tempo_events, ticks_per_quarter)
