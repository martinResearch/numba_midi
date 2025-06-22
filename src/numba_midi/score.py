"""Music score represention based on structured numpy arrays."""

from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable, Literal, Optional, overload, TYPE_CHECKING

import numpy as np

from numba_midi._score_numba import (
    _get_overlapping_notes_pairs_jit,
    extract_notes_start_stop_numba,
    get_events_program,
    get_pedals_from_controls_jit,
    get_subdivision_beat_and_bar_ticks_jit,
    recompute_tempo_times,
)
from numba_midi.instruments import (
    instrument_to_program,
    program_to_instrument,
    program_to_instrument_group,
)
from numba_midi.midi import EventArray, get_event_times, load_midi_bytes, Midi, MidiTrack, save_midi_file
from numba_midi.utils import get_tick_per_beat_array

if TYPE_CHECKING:
    from numba_midi.pianoroll import PianoRoll
NotesMode = Literal["no_overlap", "first_in_first_out", "note_off_stops_all"]

notes_mode_mapping: dict[NotesMode, int] = {
    "no_overlap": 0,
    "first_in_first_out": 1,
    "note_off_stops_all": 2,
}

note_dtype = np.dtype(
    [
        ("start", np.float64),
        ("start_tick", np.int32),
        ("duration", np.float64),
        ("duration_tick", np.int32),
        ("pitch", np.int32),
        ("velocity", np.uint8),
    ]
)

control_dtype = np.dtype([("time", np.float64), ("tick", np.int32), ("number", np.int32), ("value", np.int32)])
pedal_dtype = np.dtype(
    [("time", np.float64), ("tick", np.int32), ("duration", np.float64), ("duration_tick", np.int32)]
)
pitch_bend_dtype = np.dtype([("time", np.float64), ("tick", np.int32), ("value", np.int32)])
tempo_dtype = np.dtype([("time", np.float64), ("tick", np.int32), ("quarter_notes_per_minute", np.float64)])
signature_dtype = np.dtype(
    [
        ("time", np.float64),
        ("tick", np.int32),
        ("numerator", np.int32),
        ("denominator", np.int32),
        ("clocks_per_click", np.uint8),
        ("notated_32nd_notes_per_beat", np.uint8),
    ]
)

DRUM_CHANNEL = 9  # Midi convention


@dataclass
class Signature:
    """MIDI time signature representation."""

    time: float
    tick: int
    numerator: int
    denominator: int
    clocks_per_click: int
    notated_32nd_notes_per_beat: int

    def __post_init__(self) -> None:
        assert self.time >= 0, "Time must be non-negative"
        assert self.tick >= 0, "Tick must be non-negative"
        assert self.numerator > 0, "Numerator must be positive"
        assert self.denominator > 0, "Denominator must be positive"
        assert self.clocks_per_click > 0, "Clocks per click must be positive"
        assert self.notated_32nd_notes_per_beat > 0, "Notated 32nd notes per beat must be positive"


class SignatureArray:
    """Wrapper for a structured numpy array with signature_dtype elements."""

    def __init__(
        self,
        time: np.ndarray | list[float],
        tick: np.ndarray | list[int],
        numerator: np.ndarray | list[int],
        denominator: np.ndarray | list[int],
        clocks_per_click: np.ndarray | list[int],
        notated_32nd_notes_per_beat: np.ndarray | list[int],
    ) -> None:
        if isinstance(time, list):
            time = np.array(time, dtype=np.float64)
        if isinstance(tick, list):
            tick = np.array(tick, dtype=np.int32)
        if isinstance(numerator, list):
            numerator = np.array(numerator, dtype=np.int32)
        if isinstance(denominator, list):
            denominator = np.array(denominator, dtype=np.int32)
        if isinstance(clocks_per_click, list):
            clocks_per_click = np.array(clocks_per_click, dtype=np.int32)
        if isinstance(notated_32nd_notes_per_beat, list):
            notated_32nd_notes_per_beat = np.array(notated_32nd_notes_per_beat, dtype=np.int32)
        data = np.empty(len(time), dtype=signature_dtype)
        data["time"] = time
        data["tick"] = tick
        data["numerator"] = numerator
        data["denominator"] = denominator
        data["clocks_per_click"] = clocks_per_click
        data["notated_32nd_notes_per_beat"] = notated_32nd_notes_per_beat
        self._data = data

    @classmethod
    def from_array(cls, data: np.ndarray) -> "SignatureArray":
        if data.dtype != signature_dtype:
            raise ValueError("Invalid dtype for SignatureArray")
        if data.ndim != 1:
            raise ValueError("SignatureArray must be 1D")
        instance = cls.__new__(cls)  # Create a new instance without calling __init__
        instance._data = data
        return instance

    @classmethod
    def zeros(cls, size: int) -> "SignatureArray":
        """Initialize the SignatureArray with zeros."""
        return SignatureArray.from_array(np.zeros(size, dtype=signature_dtype))

    @property
    def time(self) -> np.ndarray:
        return self._data["time"]

    @time.setter
    def time(self, value: np.ndarray) -> None:
        self._data["time"][:] = value

    @property
    def tick(self) -> np.ndarray:
        return self._data["tick"]

    @tick.setter
    def tick(self, value: np.ndarray) -> None:
        self._data["tick"][:] = value

    @property
    def numerator(self) -> np.ndarray:
        return self._data["numerator"]

    @numerator.setter
    def numerator(self, value: np.ndarray) -> None:
        self._data["numerator"][:] = value

    @property
    def denominator(self) -> np.ndarray:
        return self._data["denominator"]

    @property
    def clocks_per_click(self) -> np.ndarray:
        return self._data["clocks_per_click"]

    @property
    def notated_32nd_notes_per_beat(self) -> np.ndarray:
        return self._data["notated_32nd_notes_per_beat"]

    @classmethod
    def concatenate(cls, arrays: Iterable["SignatureArray"]) -> "SignatureArray":
        """Concatenate multiple SignatureArrays."""
        data = np.concatenate([arr._data for arr in arrays])
        return cls.from_array(data)

    def __len__(self) -> int:
        return len(self._data)

    @overload
    def __getitem__(self, index: int) -> Signature: ...
    @overload
    def __getitem__(self, index: np.ndarray | slice | list[int]) -> "SignatureArray": ...

    def __getitem__(self, index: int | slice | np.ndarray | list[int]) -> "SignatureArray|Signature":
        if isinstance(index, int):
            result = self._data[index]
            return Signature(
                float(result["time"]),
                int(result["tick"]),
                int(result["numerator"]),
                int(result["denominator"]),
                int(result["clocks_per_click"]),
                int(result["notated_32nd_notes_per_beat"]),
            )
        result = self._data[index]
        return SignatureArray.from_array(result)

    def __iter__(self) -> Generator[Signature, None, None]:
        for i in range(len(self._data)):
            yield Signature(
                self.time[i],
                self.tick[i],
                self.numerator[i],
                self.denominator[i],
                self.clocks_per_click[i],
                self.notated_32nd_notes_per_beat[i],
            )


@dataclass
class Note:
    """MIDI note representation."""

    start: float
    start_tick: int
    duration: float
    duration_tick: int
    pitch: int
    velocity: int

    def __post_init__(self) -> None:
        assert self.start >= 0, "Start time must be non-negative. Value is {self.start}"
        assert self.start_tick >= 0, "Start tick must be non-negative. Value is {self.start_tick}"
        assert self.duration > 0, "Duration must be positive. Value is {self.duration}"
        assert self.duration_tick > 0, f"Duration tick must be positive. Value is {self.duration_tick}"
        assert 0 <= self.pitch < 128, "Pitch must be between 0 and 127"
        assert 0 <= self.velocity <= 127, "Velocity must be between 0 and 127"


class NoteArray:
    """Wrapper for a structured numpy array with note_dtype elements."""

    def __init__(
        self,
        start: np.ndarray | list[float],
        start_tick: np.ndarray | list[int],
        duration: np.ndarray | list[float],
        duration_tick: np.ndarray | list[int],
        pitch: np.ndarray | list[int],
        velocity: np.ndarray | list[int],
    ) -> None:
        if isinstance(start, list):
            start = np.array(start, dtype=np.float64)
        if isinstance(start_tick, list):
            start_tick = np.array(start_tick, dtype=np.int32)
        if isinstance(duration, list):
            duration = np.array(duration, dtype=np.float64)
        if isinstance(duration_tick, list):
            duration_tick = np.array(duration_tick, dtype=np.int32)
        if isinstance(pitch, list):
            pitch = np.array(pitch, dtype=np.int32)
        if isinstance(velocity, list):
            velocity = np.array(velocity, dtype=np.uint8)

        data = np.empty(len(start), dtype=note_dtype)
        data["start"] = start
        data["start_tick"] = start_tick
        data["duration"] = duration
        data["duration_tick"] = duration_tick
        data["pitch"] = pitch
        data["velocity"] = velocity
        self._data = data

    @classmethod
    def from_array(cls, data: np.ndarray) -> "NoteArray":
        if data.dtype != note_dtype:
            raise ValueError("Invalid dtype for NoteArray")
        if data.ndim != 1:
            raise ValueError("NoteArray must be 1D")
        instance = cls.__new__(cls)  # Create a new instance without calling __init__
        instance._data = data
        return instance

    @classmethod
    def zeros(cls, size: int) -> "NoteArray":
        """Initialize the NoteArray with zeros."""
        return NoteArray.from_array(np.zeros(size, dtype=note_dtype))

    @classmethod
    def concatenate(cls, arrays: Iterable["NoteArray"]) -> "NoteArray":
        """Concatenate multiple NoteArrays."""
        data = np.concatenate([arr._data for arr in arrays])
        return cls.from_array(data)

    @property
    def start(self) -> np.ndarray:
        return self._data["start"]

    @start.setter
    def start(self, value: np.ndarray) -> None:
        self._data["start"][:] = value

    @property
    def start_tick(self) -> np.ndarray:
        return self._data["start_tick"]

    @start_tick.setter
    def start_tick(self, value: np.ndarray) -> None:
        self._data["start_tick"][:] = value

    @property
    def duration(self) -> np.ndarray:
        return self._data["duration"]

    @duration.setter
    def duration(self, value: np.ndarray) -> None:
        self._data["duration"][:] = value

    @property
    def duration_tick(self) -> np.ndarray:
        return self._data["duration_tick"]

    @duration_tick.setter
    def duration_tick(self, value: np.ndarray) -> None:
        self._data["duration_tick"][:] = value

    @property
    def velocity(self) -> np.ndarray:
        return self._data["velocity"]

    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        self._data["velocity"][:] = value

    @property
    def pitch(self) -> np.ndarray:
        return self._data["pitch"]

    @pitch.setter
    def pitch(self, value: np.ndarray) -> None:
        self._data["pitch"][:] = value

    @overload
    def __getitem__(self, index: int) -> Note: ...
    @overload
    def __getitem__(self, index: np.ndarray | slice | list[int]) -> "NoteArray": ...

    def __getitem__(self, index: int | slice | np.ndarray | list[int]) -> "NoteArray|Note":
        if isinstance(index, int):
            result = self._data[index]
            return Note(
                float(result["start"]),
                int(result["start_tick"]),
                float(result["duration"]),
                int(result["duration_tick"]),
                int(result["pitch"]),
                int(result["velocity"]),
            )
        result = self._data[index]
        return NoteArray.from_array(result)  # Return new wrapper for slices or boolean arrays

    def delete(self, index: int | slice | np.ndarray) -> None:
        """Delete notes at the specified index."""
        if isinstance(index, np.ndarray):
            assert index.ndim == 1, "Index array must be 1D"
        new_data = np.delete(self._data, index, axis=0)
        self._data = new_data  # type: ignore

    def __setitem__(self, index: int | slice | np.ndarray, value: "NoteArray") -> None:
        self._data[index] = value._data

    def __len__(self) -> int:
        return len(self._data)

    def as_array(self) -> np.ndarray:
        return self._data

    @property
    def size(self) -> int:
        return self._data.size

    def __iter__(self) -> Generator[Note, None, None]:
        for i in range(len(self._data)):
            yield Note(
                self.start[i],
                self.start_tick[i],
                self.duration[i],
                self.duration_tick[i],
                self.pitch[i],
                self.velocity[i],
            )

    def __repr__(self) -> str:
        return f"NoteArray(size={self.size})"


@dataclass
class Control:
    """MIDI control representation."""

    time: float
    tick: int
    number: int
    value: int

    def __post_init__(self) -> None:
        assert self.time >= 0, "Time must be non-negative"
        assert self.tick >= 0, "Tick must be non-negative"
        assert 0 <= self.number < 128, "Control number must be between 0 and 127"
        assert 0 <= self.value < 128, "Control value must be between 0 and 127"


class ControlArray:
    """Wrapper for a structured numpy array with control_dtype elements."""

    def __init__(
        self,
        time: np.ndarray | list[float],
        tick: np.ndarray | list[int],
        number: np.ndarray | list[int],
        value: np.ndarray | list[int],
    ) -> None:
        if isinstance(time, list):
            time = np.array(time, dtype=np.float64)
        if isinstance(tick, list):
            tick = np.array(tick, dtype=np.int32)
        if isinstance(number, list):
            number = np.array(number, dtype=np.int32)
        if isinstance(value, list):
            value = np.array(value, dtype=np.int32)
        data = np.empty(len(time), dtype=control_dtype)
        data["time"] = time
        data["tick"] = tick
        data["number"] = number
        data["value"] = value
        self._data = data

    @classmethod
    def from_array(cls, data: np.ndarray) -> "ControlArray":
        if data.dtype != control_dtype:
            raise ValueError("Invalid dtype for ControlArray")
        if data.ndim != 1:
            raise ValueError("ControlArray must be 1D")
        instance = cls.__new__(cls)  # Create a new instance without calling __init__
        instance._data = data
        return instance

    @classmethod
    def zeros(cls, size: int) -> "ControlArray":
        """Initialize the ControlArray with zeros."""
        return ControlArray.from_array(np.zeros(size, dtype=control_dtype))

    @classmethod
    def concatenate(cls, arrays: Iterable["ControlArray"]) -> "ControlArray":
        """Concatenate multiple NoteArrays."""
        data = np.concatenate([arr._data for arr in arrays])
        return cls.from_array(data)

    @property
    def time(self) -> np.ndarray:
        return self._data["time"]

    @time.setter
    def time(self, value: np.ndarray) -> None:
        self._data["time"][:] = value

    @property
    def tick(self) -> np.ndarray:
        return self._data["tick"]

    @tick.setter
    def tick(self, value: np.ndarray) -> None:
        self._data["tick"][:] = value

    @property
    def number(self) -> np.ndarray:
        return self._data["number"]

    @number.setter
    def number(self, value: np.ndarray) -> None:
        self._data["number"][:] = value

    @property
    def value(self) -> np.ndarray:
        return self._data["value"]

    @value.setter
    def value(self, value: np.ndarray) -> None:
        self._data["value"][:] = value

    @overload
    def __getitem__(self, index: int) -> Control: ...
    @overload
    def __getitem__(self, index: np.ndarray | slice | list[int]) -> "ControlArray": ...

    def __getitem__(self, index: int | slice | np.ndarray | list[int]) -> "ControlArray|Control":
        if isinstance(index, int):
            result = self._data[index]
            return Control(
                float(result["time"]),
                int(result["tick"]),
                int(result["number"]),
                int(result["value"]),
            )
        result = self._data[index]
        return ControlArray.from_array(result)  # Return new wrapper for slices or boolean arrays

    def __setitem__(self, index: int | slice | np.ndarray, value: "ControlArray") -> None:
        self._data[index] = value._data

    def __len__(self) -> int:
        return len(self._data)

    def as_array(self) -> np.ndarray:
        return self._data

    @property
    def size(self) -> int:
        return self._data.size

    def __iter__(self) -> Generator[Control, None, None]:
        for i in range(len(self._data)):
            yield Control(
                self.time[i],
                self.tick[i],
                self.number[i],
                self.value[i],
            )

    def sort_time(self) -> None:
        """Sort the ControlArray by time."""
        sorted_indices = np.argsort(self.time)
        self._data = self._data[sorted_indices]  # type: ignore


@dataclass
class Pedal:
    """MIDI pedal representation."""

    time: float
    tick: int
    duration: float
    duration_tick: int

    def __post_init__(self) -> None:
        assert self.time >= 0, "Time must be non-negative"
        assert self.tick >= 0, "Tick must be non-negative"
        assert self.duration > 0, "Duration must be positive"
        assert self.duration_tick > 0, "Duration tick must be positive"


class PedalArray:
    """Wrapper for a structured numpy array with pedal_dtype elements."""

    def __init__(
        self,
        time: np.ndarray | list[float],
        tick: np.ndarray | list[int],
        duration: np.ndarray | list[float],
    ) -> None:
        if isinstance(time, list):
            time = np.array(time, dtype=np.float64)
        if isinstance(tick, list):
            tick = np.array(tick, dtype=np.int32)
        if isinstance(duration, list):
            duration = np.array(duration, dtype=np.float64)
        data = np.empty(len(time), dtype=note_dtype)
        data["time"] = time
        data["tick"] = tick
        data["duration"] = duration
        self._data = data

    @classmethod
    def from_array(cls, data: np.ndarray) -> "PedalArray":
        if data.dtype != pedal_dtype:
            raise ValueError("Invalid dtype for ControlArray")
        if data.ndim != 1:
            raise ValueError("ControlArray must be 1D")
        instance = cls.__new__(cls)  # Create a new instance without calling __init__
        instance._data = data
        return instance

    @classmethod
    def zeros(cls, size: int) -> "PedalArray":
        """Initialize the PedalArray with zeros."""
        return PedalArray.from_array(np.zeros(size, dtype=pedal_dtype))

    @classmethod
    def concatenate(cls, arrays: Iterable["PedalArray"]) -> "PedalArray":
        """Concatenate multiple PedalArrays."""
        data = np.concatenate([arr._data for arr in arrays])
        return cls.from_array(data)

    @property
    def time(self) -> np.ndarray:
        return self._data["time"]

    @time.setter
    def time(self, value: np.ndarray) -> None:
        self._data["time"][:] = value

    @property
    def tick(self) -> np.ndarray:
        return self._data["tick"]

    @tick.setter
    def tick(self, value: np.ndarray) -> None:
        self._data["tick"][:] = value

    @property
    def duration(self) -> np.ndarray:
        return self._data["duration"]

    @duration.setter
    def duration(self, value: np.ndarray) -> None:
        self._data["duration"][:] = value

    @property
    def duration_tick(self) -> np.ndarray:
        return self._data["duration_tick"]

    @duration_tick.setter
    def duration_tick(self, value: np.ndarray) -> None:
        self._data["duration_tick"][:] = value

    @overload
    def __getitem__(self, index: int) -> Pedal: ...
    @overload
    def __getitem__(self, index: np.ndarray | slice | list[int]) -> "PedalArray": ...

    def __getitem__(self, index: int | slice | np.ndarray | list[int]) -> "PedalArray|Pedal":
        if isinstance(index, int):
            result = self._data[index]
            return Pedal(
                float(result["time"]),
                int(result["tick"]),
                float(result["duration"]),
                int(result["duration_tick"]),
            )
        result = self._data[index]
        return PedalArray.from_array(result)  # Return new wrapper for slices or boolean arrays

    def __setitem__(self, index: int | slice | np.ndarray, value: "PedalArray") -> None:
        self._data[index] = value._data

    def __len__(self) -> int:
        return len(self._data)

    def as_array(self) -> np.ndarray:
        return self._data

    @property
    def size(self) -> int:
        return self._data.size

    def __iter__(self) -> Generator[Pedal, None, None]:
        for i in range(len(self._data)):
            yield Pedal(
                self.time[i],
                self.tick[i],
                self.duration[i],
                self.duration_tick[i],
            )


@dataclass
class PitchBend:
    """MIDI pitch bend representation."""

    time: float
    tick: int
    value: int

    def __post_init__(self) -> None:
        assert self.time >= 0, "Time must be non-negative"
        assert self.tick >= 0, "Tick must be non-negative"
        assert -8192 <= self.value <= 8191, "Pitch bend value must be between -8192 and 8191"


class PitchBendArray:
    """Wrapper for a structured numpy array with pitch_bend_dtype elements."""

    def __init__(
        self,
        time: np.ndarray | list[float],
        tick: np.ndarray | list[int],
        value: np.ndarray | list[int],
    ) -> None:
        if isinstance(time, list):
            time = np.array(time, dtype=np.float64)
        if isinstance(tick, list):
            tick = np.array(tick, dtype=np.int32)
        if isinstance(value, list):
            value = np.array(value, dtype=np.int32)
        data = np.empty(len(time), dtype=note_dtype)
        data["time"] = time
        data["tick"] = tick
        data["value"] = value
        self._data = data

    @classmethod
    def from_array(cls, data: np.ndarray) -> "PitchBendArray":
        if data.dtype != pitch_bend_dtype:
            raise ValueError("Invalid dtype for ControlArray")
        if data.ndim != 1:
            raise ValueError("ControlArray must be 1D")
        instance = cls.__new__(cls)  # Create a new instance without calling __init__
        instance._data = data
        return instance

    @classmethod
    def zeros(cls, size: int) -> "PitchBendArray":
        """Initialize the PitchBendArray with zeros."""
        return PitchBendArray.from_array(np.zeros(size, dtype=pitch_bend_dtype))

    @classmethod
    def concatenate(cls, arrays: Iterable["PitchBendArray"]) -> "PitchBendArray":
        """Concatenate multiple PitchBendArrays."""
        data = np.concatenate([arr._data for arr in arrays])
        return cls.from_array(data)

    @property
    def time(self) -> np.ndarray:
        return self._data["time"]

    @time.setter
    def time(self, value: np.ndarray) -> None:
        self._data["time"][:] = value

    @property
    def tick(self) -> np.ndarray:
        return self._data["tick"]

    @tick.setter
    def tick(self, value: np.ndarray) -> None:
        self._data["tick"][:] = value

    @property
    def value(self) -> np.ndarray:
        return self._data["value"]

    @value.setter
    def value(self, value: np.ndarray) -> None:
        self._data["value"][:] = value

    @overload
    def __getitem__(self, index: int) -> PitchBend: ...
    @overload
    def __getitem__(self, index: np.ndarray | slice | list[int]) -> "PitchBendArray": ...

    def __getitem__(self, index: int | slice | np.ndarray | list[int]) -> "PitchBendArray|PitchBend":
        if isinstance(index, int):
            result = self._data[index]
            return PitchBend(
                float(result["time"]),
                int(result["tick"]),
                int(result["value"]),
            )
        result = self._data[index]
        return PitchBendArray.from_array(result)  # Return new wrapper for slices or boolean arrays

    def __setitem__(self, index: int | slice | np.ndarray, value: "PitchBendArray") -> None:
        self._data[index] = value._data

    def __len__(self) -> int:
        return len(self._data)

    def as_array(self) -> np.ndarray:
        return self._data

    @property
    def size(self) -> int:
        return self._data.size

    def __iter__(self) -> Generator[PitchBend, None, None]:
        for i in range(len(self._data)):
            yield PitchBend(
                self.time[i],
                self.tick[i],
                self.value[i],
            )


@dataclass
class Tempo:
    """MIDI tempo representation."""

    time: float
    tick: int
    quarter_notes_per_minute: float

    def __post_init__(self) -> None:
        assert self.time >= 0, "Time must be non-negative"
        assert self.tick >= 0, "Tick must be non-negative"
        assert self.quarter_notes_per_minute > 0, "QNPM must be positive"

    def __repr__(self) -> str:
        return f"Tempo(time={self.time}, tick={self.tick}, quarter_notes_per_minute={self.quarter_notes_per_minute})"


class TempoArray:
    """Wrapper for a structured numpy array with tempo_dtype elements."""

    def __init__(
        self,
        time: np.ndarray | list[float],
        tick: np.ndarray | list[int],
        quarter_notes_per_minute: np.ndarray | list[float],
    ) -> None:
        if isinstance(time, list):
            time = np.array(time, dtype=np.float64)
        if isinstance(tick, list):
            tick = np.array(tick, dtype=np.int32)
        if isinstance(quarter_notes_per_minute, list):
            quarter_notes_per_minute = np.array(quarter_notes_per_minute, dtype=np.float64)
        data = np.empty(len(time), dtype=tempo_dtype)
        data["time"] = time
        data["tick"] = tick
        data["quarter_notes_per_minute"] = quarter_notes_per_minute
        self._data = data

    @classmethod
    def from_array(cls, data: np.ndarray) -> "TempoArray":
        if data.dtype != tempo_dtype:
            raise ValueError("Invalid dtype for ControlArray")
        if data.ndim != 1:
            raise ValueError("ControlArray must be 1D")
        instance = cls.__new__(cls)  # Create a new instance without calling __init__
        instance._data = data
        return instance

    @classmethod
    def zeros(cls, size: int) -> "TempoArray":
        """Initialize the TempoArray with zeros."""
        return TempoArray.from_array(np.zeros(size, dtype=tempo_dtype))

    @classmethod
    def concatenate(cls, arrays: Iterable["TempoArray"]) -> "TempoArray":
        """Concatenate multiple TempoArrays."""
        data = np.concatenate([arr._data for arr in arrays])
        return cls.from_array(data)

    @property
    def time(self) -> np.ndarray:
        return self._data["time"]

    @time.setter
    def time(self, value: np.ndarray) -> None:
        self._data["time"][:] = value

    @property
    def tick(self) -> np.ndarray:
        return self._data["tick"]

    @tick.setter
    def tick(self, value: np.ndarray) -> None:
        self._data["tick"][:] = value

    @property
    def quarter_notes_per_minute(self) -> np.ndarray:
        return self._data["quarter_notes_per_minute"]

    @quarter_notes_per_minute.setter
    def quarter_notes_per_minute(self, value: np.ndarray) -> None:
        self._data["quarter_notes_per_minute"][:] = value

    @overload
    def __getitem__(self, index: int) -> Tempo: ...
    @overload
    def __getitem__(self, index: np.ndarray | slice | list[int]) -> "TempoArray": ...

    def __getitem__(self, index: int | slice | np.ndarray | list[int]) -> "TempoArray|Tempo":
        if isinstance(index, int):
            result = self._data[index]
            return Tempo(
                float(result["time"]),
                int(result["tick"]),
                float(result["quarter_notes_per_minute"]),
            )
        result = self._data[index]
        return TempoArray.from_array(result)  # Return new wrapper for slices or boolean arrays

    def __setitem__(self, index: int | slice | np.ndarray, value: "TempoArray") -> None:
        self._data[index] = value._data

    def __len__(self) -> int:
        return len(self._data)

    def as_array(self) -> np.ndarray:
        return self._data

    @property
    def size(self) -> int:
        return self._data.size

    def __iter__(self) -> Generator[Tempo, None, None]:
        for i in range(len(self._data)):
            yield Tempo(self.time[i], self.tick[i], self.quarter_notes_per_minute[i])

    def recompute_times(self, ticks_per_quarter: int) -> None:
        recompute_tempo_times(self._data, ticks_per_quarter)


@dataclass
class Track:
    """MIDI track representation."""

    program: int
    is_drum: bool
    name: str
    notes: NoteArray
    controls: ControlArray
    pedals: PedalArray
    pitch_bends: PitchBendArray
    channel: Optional[int] = None
    midi_track_id: Optional[int] = None

    def __post_init__(self) -> None:
        assert isinstance(self.notes, NoteArray), "Notes must be a structured numpy array with note_dtype elements"
        assert isinstance(self.controls, ControlArray), (
            "Controls must be a structured numpy array with control_dtype elements"
        )
        assert isinstance(self.pedals, PedalArray), "Pedals must be a structured numpy array with pedal_dtype elements"
        assert isinstance(self.pitch_bends, PitchBendArray), (
            "Pitch bends must be a structured numpy array with pitch_bend_dtype elements"
        )

    @classmethod
    def empty(cls, program: int, is_drum: bool, name: str) -> "Track":
        """Create an empty track."""
        return Track(
            program=program,
            is_drum=is_drum,
            name=name,
            notes=NoteArray.zeros(0),
            controls=ControlArray.zeros(0),
            pedals=PedalArray.zeros(0),
            pitch_bends=PitchBendArray.zeros(0),
        )

    def last_tick(self) -> int:
        """Get the last tick of the track."""
        last_tick = 0
        if len(self.notes) > 0:
            last_tick = np.max(self.notes.start_tick + self.notes.duration_tick)
        if len(self.controls) > 0:
            last_tick = max(last_tick, np.max(self.controls.tick))
        if len(self.pedals) > 0:
            last_tick = max(last_tick, np.max(self.pedals.tick))
        if len(self.pitch_bends) > 0:
            last_tick = max(last_tick, np.max(self.pitch_bends.tick))

        return last_tick

    def __repr__(self) -> str:
        return (
            f"Track {self.name} with {len(self.notes)} notes, program={self.program}, "
            f"{len(self.controls)} controls, {len(self.pedals)} pedals, {len(self.pitch_bends)} pitch bends"
        )


@dataclass
class Score:
    """MIDI score representation."""

    tracks: list[Track]
    last_tick: int
    tempo: TempoArray  # 1D structured numpy array with tempo_dtype elements
    time_signature: SignatureArray  # 1D structured numpy array with signature_dtype elements
    lyrics: list[tuple[int, str]] | None = None
    ticks_per_quarter: int = 480

    @property
    def num_notes(self) -> int:
        """Get the number of notes in the score."""
        num_notes = sum(len(track.notes) for track in self.tracks)
        return num_notes

    @property
    def num_tracks(self) -> int:
        """Get the number of tracks in the score."""
        return len(self.tracks)

    def __repr__(self) -> str:
        return f"Score(num_tracks={self.num_tracks}, num_notes={self.num_notes}, duration={self.duration:02g})"

    @property
    def duration(self) -> float:
        return self.tick_to_time(self.last_tick)

    @duration.setter
    def duration(self, value: float) -> None:
        """Set the duration of the score."""
        self.last_tick = int(np.round(self.time_to_tick(value)))

    def __post_init__(self) -> None:
        assert isinstance(self.tempo, TempoArray), "Tempo must be a structured numpy array with tempo_dtype elements"
        assert isinstance(self.time_signature, SignatureArray), (
            "Time signature must be a structured numpy array with signature_dtype elements"
        )
        assert self.tracks is not None, "Tracks must be a list of Track objects"
        # assert len(self.tracks) > 0, "Tracks must be a non-empty list of Track objects"
        assert self.last_tick > 0, "Duration must be a positive float"
        assert len(self.tempo) > 0, "Tempo must a non-empty"

    def to_pianoroll(
        self,
        time_step: float,
        pitch_min: int,
        pitch_max: int,
        num_bin_per_semitone: int,
        shorten_notes: bool = True,
        antialiasing: bool = False,
    ) -> "PianoRoll":
        from numba_midi.pianoroll import score_to_piano_roll

        return score_to_piano_roll(
            self,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            time_step=time_step,
            num_bin_per_semitone=num_bin_per_semitone,
            shorten_notes=shorten_notes,
            antialiasing=antialiasing,
        )

    def get_subdivision_beat_and_bar_ticks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return get_subdivision_beat_and_bar_ticks_jit(self.ticks_per_quarter, self.last_tick, self.time_signature._data)

    def get_subdivision_beat_and_bar_times(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the beat and bar times in seconds."""
        beat_ticks, bar_ticks, subdivision_ticks = self.get_subdivision_beat_and_bar_ticks()
        bar_time = ticks_to_times(bar_ticks, self.tempo, self.ticks_per_quarter)
        beat_time = ticks_to_times(beat_ticks, self.tempo, self.ticks_per_quarter)
        subdivision_time = ticks_to_times(subdivision_ticks, self.tempo, self.ticks_per_quarter)
        return beat_time, bar_time, subdivision_time

    def save(self, file_path: str | Path) -> None:
        """Save the score to a MIDI file."""
        save_score_to_midi(self, str(file_path))

    @classmethod
    def load(
        cls,
        file_path: str | Path,
        notes_mode: NotesMode = "note_off_stops_all",
        minimize_tempo: bool = True,
        check_round_trip: bool = False,
    ) -> "Score":
        """Load a score from a MIDI file."""
        return load_score(
            str(file_path), notes_mode=notes_mode, minimize_tempo=minimize_tempo, check_round_trip=check_round_trip
        )

    def time_to_tick(self, time: float) -> int:
        return time_to_tick(time, self.tempo, self.ticks_per_quarter)

    def time_to_float_tick(self, time: float) -> float:
        """Convert time to float tick."""
        return time_to_float_tick(time, self.tempo, self.ticks_per_quarter)

    def times_to_ticks(self, times: np.ndarray) -> np.ndarray:
        return times_to_ticks(times, self.tempo, self.ticks_per_quarter)

    def tick_to_time(self, tick: float) -> float:
        return tick_to_time(tick, self.tempo, self.ticks_per_quarter)

    def ticks_to_times(self, ticks: np.ndarray) -> np.ndarray:
        return ticks_to_times(ticks, self.tempo, self.ticks_per_quarter)

    def round_to_tick(self, time: float) -> tuple[float, int]:
        """Round the time to the nearest tick."""
        rounded_time, tick = round_to_tick(time, self.tempo, self.ticks_per_quarter)
        return rounded_time, tick

    def round_to_ticks(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Round the given times to the nearest tick."""
        rounded_times, ticks = round_to_ticks(times, self.tempo, self.ticks_per_quarter)
        return rounded_times, ticks

    def create_notes(
        self,
        start: np.ndarray,
        pitch: np.ndarray,
        duration: np.ndarray,
        velocity: np.ndarray,
    ) -> NoteArray:
        """Create notes from start, pitch and duration."""
        assert len(start) == len(pitch) == len(duration), "start, pitch and duration must have the same length"

        notes = NoteArray.zeros(len(start))
        start, start_ticks = self.round_to_ticks(start)
        end = start + duration
        end, end_ticks = self.round_to_ticks(end)
        duration_ticks = end_ticks - start_ticks
        duration = end - start
        notes.start = start
        notes.start_tick = start_ticks
        notes.duration = duration
        notes.duration_tick = duration_ticks
        notes.pitch = pitch
        notes.velocity = velocity
        return notes

    def create_note(self, start: float, pitch: int, duration: float, velocity: int) -> NoteArray:
        """Create a note from start, pitch and duration."""
        notes = NoteArray.zeros(1)
        start, start_tick = self.round_to_tick(start)
        end = start + duration
        end, end_ticks = self.round_to_tick(end)
        duration_ticks = end_ticks - start_tick
        duration = end - start
        notes.start[:] = start
        notes.start_tick[:] = start_tick
        notes.duration[:] = duration
        notes.duration_tick[:] = duration_ticks
        notes.pitch[:] = pitch
        notes.velocity[:] = velocity
        return notes

    def without_empty_tracks(self) -> "Score":
        return remove_empty_tracks(self)

    def crop(self, start: float, end: float) -> "Score":
        """Crop the score to the given start and end time."""
        return crop_score(self, start, end)

    def select_tracks(self, track_ids: list[int]) -> "Score":
        return select_tracks(self, track_ids)

    def add_track(self, track: Track) -> None:
        """Add a track to the score."""
        assert isinstance(track, Track), "Track must be a Track object"
        self.tracks.append(track)

    def remove_track(self, track_id: int) -> None:
        """Delete a track from the score."""
        assert 0 <= track_id < len(self.tracks), "Track ID out of range"
        del self.tracks[track_id]

    def add_empty_track(self, name: str, program: int, is_drum: bool = False) -> None:
        """Add an empty track to the score."""
        assert 0 <= program < 128, "Program must be between 0 and 127"
        notes = NoteArray.zeros(0)
        controls = ControlArray.zeros(0)
        pedals = PedalArray.zeros(0)
        pitch_bends = PitchBendArray.zeros(0)
        track = Track(program, is_drum, name, notes, controls, pedals, pitch_bends)
        self.tracks.append(track)

    def add_notes(
        self, track_id: int, time: np.ndarray, duration: np.ndarray, pitch: np.ndarray, velocity: np.ndarray
    ) -> None:
        """Add notes to the score."""
        assert len(time) == len(duration) == len(pitch) == len(velocity), (
            "time, pitch and valocity must have the same length"
        )
        end = time + duration
        start, start_tick = self.round_to_ticks(time)
        end, end_ticks = self.round_to_ticks(end)
        duration_ticks = end_ticks - start_tick
        notes = NoteArray.zeros(len(time))
        notes.start = start
        notes.duration = duration
        notes.start_tick = start_tick
        notes.duration_tick = duration_ticks
        notes.pitch = pitch
        notes.velocity = velocity
        track = self.tracks[track_id]
        track.notes = NoteArray.concatenate((track.notes, notes))

    def add_controls(self, track_id: int, time: np.ndarray, number: np.ndarray, value: np.ndarray) -> None:
        """Add controls to the score."""
        assert len(time) == len(number) == len(value), "time, number and value must have the same length"
        time, ticks = self.round_to_ticks(time)
        controls = ControlArray.zeros(len(time))
        controls.time = time
        controls.tick = ticks
        controls.number = number
        controls.value = value
        track = self.tracks[track_id]

        if track.controls.size > 0:
            # Concatenate the underlying arrays and create a new ControlArray
            track.controls = ControlArray.concatenate((track.controls, controls))
        else:
            track.controls = controls

    def add_pitch_bends(self, track_id: int, time: np.ndarray, value: np.ndarray) -> None:
        """Add pitch bends to the score."""
        assert len(time) == len(value), "time and value must have the same length"
        time, ticks = self.round_to_ticks(time)
        pitch_bends = PitchBendArray.zeros(len(time))
        pitch_bends.time = time
        pitch_bends.tick = ticks
        pitch_bends.value = value
        track = self.tracks[track_id]

        if track.pitch_bends.size > 0:
            track.pitch_bends = PitchBendArray.concatenate((track.pitch_bends, pitch_bends))
        else:
            track.pitch_bends = pitch_bends

    def add_tempos(self, time: np.ndarray, quarter_notes_per_minute: np.ndarray) -> None:
        """Add tempos to the score."""
        assert len(time) == len(quarter_notes_per_minute), "time and quarter_notes_per_minute must have the same length"
        (
            time,
            ticks,
        ) = self.round_to_ticks(time)

        tempos = TempoArray.zeros(len(time))
        tempos.time = time
        tempos.tick = ticks
        tempos.quarter_notes_per_minute = quarter_notes_per_minute
        tempos = TempoArray.concatenate((self.tempo, tempos))
        self.tempo = tempos

        self._resort_tempo()
        self._recompute_times()

    def remove_tempos(self, mask: np.ndarray) -> None:
        self.tempo = self.tempo[~mask]
        self._recompute_times()

    def _resort_tempo(self) -> None:
        """Resort the tempo array."""
        order = np.argsort(self.tempo.tick)
        self.tempo = self.tempo[order]

    def _recompute_times(self) -> None:
        """Recompute the times of the score keeping ticks fixes."""
        self.tempo.recompute_times(self.ticks_per_quarter)
        for track in self.tracks:
            track.notes.start = self.ticks_to_times(track.notes.start_tick)
            notes_end_ticks = track.notes.start_tick + track.notes.duration_tick
            note_end = self.ticks_to_times(notes_end_ticks)
            durations = note_end - track.notes.start
            track.notes.duration = durations
            track.controls.time = self.ticks_to_times(track.controls.tick)
            track.pedals.time = self.ticks_to_times(track.pedals.tick)
            track.pitch_bends.time = self.ticks_to_times(track.pitch_bends.tick)

    def times_to_beats(self, time: np.ndarray) -> np.ndarray:
        """Convert time to beats."""
        assert len(time) > 0, "Time must be a non-empty array"
        # Compute the beat positions in seconds using the tempo
        _, beat_ticks, _ = self.get_subdivision_beat_and_bar_ticks()
        ticks = self.times_to_ticks(time)
        # Compute the  positions in beats
        beat_idx = np.searchsorted(beat_ticks, ticks, side="right") - 1
        signature_idx = np.searchsorted(self.time_signature.time, time, side="right") - 1
        ticks_per_beat = get_tick_per_beat_array(
            self.ticks_per_quarter,
            self.time_signature.numerator[signature_idx],
            self.time_signature.denominator[signature_idx],
        )
        beats = beat_idx + (ticks - beat_ticks[beat_idx]) / ticks_per_beat
        return beats

    def time_to_beat(self, time: float) -> float:
        """Convert time to beat."""
        assert time >= 0, "Time must be non-negative"
        return float(self.times_to_beats(np.array([time]))[0])

    def beats_to_ticks(self, beats: np.ndarray) -> np.ndarray:
        _, beat_ticks, _ = self.get_subdivision_beat_and_bar_ticks()
        beats_floor = np.floor(beats).astype(np.int32)
        signature_idx = np.searchsorted(self.time_signature.time, beats, side="right") - 1
        ticks_per_beat = get_tick_per_beat_array(
            self.ticks_per_quarter,
            self.time_signature.numerator[signature_idx],
            self.time_signature.denominator[signature_idx],
        )
        return beat_ticks[beats_floor] + (beats - beats_floor) * ticks_per_beat

    def ticks_to_quarter_notes(self, ticks: np.ndarray) -> np.ndarray:
        """Convert ticks to quarter notes."""
        assert len(ticks) > 0, "Ticks must be a non-empty array"
        # Compute the quarter note positions in seconds using the tempo
        quarter_note_ticks = self.ticks_per_quarter
        # Compute the positions in quarter notes
        return ticks / quarter_note_ticks

    def quarter_notes_to_ticks(self, quarter_notes: np.ndarray) -> np.ndarray:
        """Convert quarter notes to ticks."""
        assert len(quarter_notes) > 0, "Quarter notes must be a non-empty array"
        # Compute the quarter note positions in seconds using the tempo
        quarter_note_ticks = self.ticks_per_quarter
        # Compute the positions in ticks
        return quarter_notes * quarter_note_ticks

    def quarter_notes_to_times(self, quarter_notes: np.ndarray) -> np.ndarray:
        """Convert quarter notes to time."""
        assert len(quarter_notes) > 0, "Quarter notes must be a non-empty array"
        # Compute the positions in time
        return self.ticks_to_times(self.quarter_notes_to_ticks(quarter_notes))

    def quarter_note_to_time(self, quarter_note: float) -> float:
        """Convert a quarter note to time."""
        assert quarter_note >= 0, "Quarter note must be non-negative"
        return float(self.quarter_notes_to_times(np.array([quarter_note]))[0])

    def times_to_quarter_notes(self, times: np.ndarray) -> np.ndarray:
        """Convert times to quarter notes."""
        assert times.ndim == 1, "Input must be a 1D array"
        return self.ticks_to_quarter_notes(self.times_to_ticks(times).astype(np.float32))

    def time_to_quarter_note(self, time: float) -> float:
        """Convert time to quarter note."""
        assert time >= 0, "Time must be non-negative"
        return float(self.times_to_quarter_notes(np.array([time]))[0])

    def beats_to_times(self, beats: np.ndarray) -> np.ndarray:
        """Convert beats to time."""
        return self.ticks_to_times(self.beats_to_ticks(beats))

    def beat_to_time(self, beat: float) -> float:
        """Convert a beat to time."""
        return self.tick_to_time(self.beat_to_tick(beat))

    def beat_to_tick(self, beat: float) -> int:
        """Convert a beat to tick."""
        return int(self.beats_to_ticks(np.array([beat]))[0])

    def quantize_times(self, times: np.ndarray, subdivision: int) -> np.ndarray:
        """Quantize the score to the given time subdivision factor."""
        assert subdivision > 0, "Subdivision must be positive"
        quarter_notes = self.times_to_quarter_notes(times)
        # Quantize the beats to the nearest step
        quantized_quarter_notes = np.round(quarter_notes * subdivision / 4) * (4 / subdivision)
        # Convert the quantized beats back to time
        quantized_times = self.quarter_notes_to_times(quantized_quarter_notes)
        return quantized_times

    def quantize_time(self, time: float, subdivision: int) -> float:
        """Quantize the score to the notes subdivision factor.

        subdivision==4 means quarter notes
        subdivision==8 means eighth notes
        subdivision==16 means sixteenth notes
        """
        return float(self.quantize_times(np.array([time]), subdivision)[0])

    def to_midi(self) -> Midi:
        """Convert the Score to a Midi."""
        return score_to_midi(self)


def group_data(keys: list[np.ndarray], data: Optional[np.ndarray] = None) -> dict[Any, np.ndarray]:
    """Group data by keys."""
    order = np.lexsort(keys)
    # make sure the keys have the same length
    for key_array in keys:
        assert len(key_array) > 0, "Keys must have at least one element"
        assert len(key_array) == len(keys[0]), "Keys must have the same length"
    is_boundary = np.zeros((len(keys[0]) - 1), dtype=bool)
    for i in range(len(keys)):
        is_boundary |= np.diff(keys[i][order]) != 0
    boundaries = np.nonzero(is_boundary)[0]
    starts = np.concatenate(([0], boundaries + 1))
    ends = np.concatenate((boundaries + 1, [len(keys[0])]))
    output: dict[tuple, np.ndarray] = {}
    for i in range(len(starts)):
        key = tuple([key_array[order[starts[i]]] for key_array in keys])
        if len(keys) == 1:
            key = key[0]
        if data is not None:
            output[key] = data[order[starts[i] : ends[i]]]
        else:
            output[key] = order[starts[i] : ends[i]]
    return output


def extract_notes_start_stop(note_events: EventArray, notes_mode: NotesMode) -> tuple[np.ndarray, np.ndarray]:
    notes_order = np.lexsort((note_events.event_type, note_events.tick, note_events.value1, note_events.channel))
    sorted_note_events = note_events[notes_order]
    ordered_note_start_ids, ordered_note_stop_ids = extract_notes_start_stop_numba(
        sorted_note_events._data, notes_mode_mapping[notes_mode]
    )
    if len(ordered_note_start_ids) > 0:
        note_start_ids = notes_order[ordered_note_start_ids]
        note_stop_ids = notes_order[ordered_note_stop_ids]
        # restore order to mach the one in the original midi file
        order = np.argsort(note_start_ids)
        note_start_ids = note_start_ids[order]
        note_stop_ids = note_stop_ids[order]
    else:
        note_start_ids = np.zeros((0,), dtype=np.int32)
        note_stop_ids = np.zeros((0,), dtype=np.int32)

    return note_start_ids, note_stop_ids


def get_pedals_from_controls(channel_controls: ControlArray) -> PedalArray:
    """Get pedal events from control changes."""
    pedals_start, pedals_end = get_pedals_from_controls_jit(channel_controls.as_array())
    if len(pedals_start) > 0:
        pedals = PedalArray.zeros(len(pedals_start))
        pedals.time = channel_controls.time[pedals_start]
        pedals.tick = channel_controls.tick[pedals_start]
        pedals.duration = channel_controls.time[pedals_end] - channel_controls.time[pedals_start]
        pedals.duration_tick = channel_controls.tick[pedals_end] - channel_controls.tick[pedals_start]
    else:
        pedals = PedalArray.zeros(0)
    return pedals


def midi_to_score(midi_score: Midi, minimize_tempo: bool = True, notes_mode: NotesMode = "note_off_stops_all") -> Score:
    """Convert a MidiScore to a Score.

    Convert from event-based representation notes with durations
    """
    tracks = []
    duration_tick = 0
    # assert len(midi_score.tracks) == 1, "Only one track is supported for now"
    ticks_per_quarter = midi_score.ticks_per_quarter
    all_tempo_events = []

    for midi_track in midi_score.tracks:
        tempo_change_mask = midi_track.events.event_type == 5
        num_tempo_change = np.sum(tempo_change_mask)
        if num_tempo_change > 0:
            tempo_change = midi_track.events[tempo_change_mask]
            # keep only the last tempo change for each tick
            keep = np.hstack((np.diff(tempo_change.tick) > 0, [True]))
            all_tempo_events.append(tempo_change[keep])
    if len(all_tempo_events) > 0:
        tempo_events = EventArray.concatenate(all_tempo_events)
        # sort by tick
        tempo_events = tempo_events[np.argsort(tempo_events.tick)]
        # keep only the last tempo change for each tick
        tempo_events = tempo_events[np.hstack((np.diff(tempo_events.tick) > 0, [True]))]
    else:
        # if no tempo events are found, we create a default one
        tempo_events = EventArray.zeros(1)
        tempo_events.event_type[:] = 5
        tempo_events.channel[:] = 0
        tempo_events.value1[:] = 120 * 1000000.0 / 60.0
        tempo_events.value2[:] = 0
        tempo_events.tick[:] = 0

    tempo_events_times = get_event_times(tempo_events._data, tempo_events._data, midi_score.ticks_per_quarter)

    # tempo event value1 correspond to the tempo in microseconds per quarter note
    # convert
    tempo = TempoArray(
        time=tempo_events_times,
        tick=tempo_events.tick,
        quarter_notes_per_minute=60000000 / tempo_events.value1,
    )

    # get all the signature events
    all_signature_events = []
    for midi_track in midi_score.tracks:
        signature_events = midi_track.events[midi_track.events.event_type == 10]
        if signature_events.size > 0:
            all_signature_events.append(signature_events)

    if len(all_signature_events) > 0:
        signature_events = EventArray.concatenate(all_signature_events)
        # sort by tick
        signature_events = signature_events[np.argsort(signature_events.tick)]
        # keep only the last signature change for each tick
        signature_events = signature_events[np.hstack((np.diff(signature_events.tick) > 0, [True]))]
    else:
        signature_events = EventArray.zeros(1)
        signature_events.value1[:] = 4
        signature_events.value2[:] = 2

    signature_events_times = get_event_times(signature_events._data, tempo_events._data, midi_score.ticks_per_quarter)
    signature = SignatureArray(
        time=signature_events_times,
        tick=signature_events.tick,
        numerator=signature_events.value1,
        denominator=2**signature_events.value2,
        clocks_per_click=signature_events.value3,
        notated_32nd_notes_per_beat=signature_events.value4,
    )
    # remove unnecessary tempo events
    if minimize_tempo:
        tempo = tempo[np.hstack(([True], (np.diff(tempo.quarter_notes_per_minute) != 0)))]

    lyrics = []
    for _, midi_track in enumerate(midi_score.tracks):
        if midi_track.lyrics is not None:
            lyrics.extend(midi_track.lyrics)
    # sort the lyrics by tick
    lyrics = sorted(lyrics, key=lambda x: x[0])

    for midi_track_id, midi_track in enumerate(midi_score.tracks):
        if midi_track.events.size == 0:
            continue

        # get the program for each event
        events_programs = get_events_program(midi_track.events._data)

        events = midi_track.events
        # compute the tick and time of each event
        events_ticks = events.tick
        events_times = get_event_times(events._data, tempo_events._data, midi_score.ticks_per_quarter)

        duration_tick = max(duration_tick, events.tick.max())

        # if only tempo events, skip the track
        if np.all(midi_track.events.event_type == 5):
            continue

        # remove end of track, tempy and signature events as they are not associated to a channel
        keep = (events.event_type != 9) & (events.event_type != 5) & (events.event_type != 10)
        events_programs = events_programs[keep]
        events = events[keep]
        events_ticks = events_ticks[keep]
        events_times = events_times[keep]

        if len(events_times) == 0:
            continue
        # sort all the events in lexicographic order by channel and tick
        # this allows to have a order for the events that simplifies the code to process them
        events_groups = group_data([events_programs, events.channel])
        # sort in lexicographic order by pitch first and then by tick, then even type
        # this allows to have an  order for the events that simplifies the logic to
        # extract matching note starts and stop events.
        # We sort by inverse of event type in order to deal with the case there is no gap
        # between two consecutive notes
        # extract the event of type note on or note off
        notes_events_ids = np.nonzero((events.event_type == 0) | (events.event_type == 1))[0]
        note_groups: dict[Any, np.ndarray] = {}
        if len(notes_events_ids) > 0:
            note_events = events[notes_events_ids]

            note_start_ids, note_stop_ids = extract_notes_start_stop(note_events, notes_mode)
            assert np.all(np.diff(note_start_ids) >= 0), "note start ids should be sorted"
            if note_start_ids.size > 0:
                note_start_ids = notes_events_ids[note_start_ids]
                note_stop_ids = notes_events_ids[note_stop_ids]

                # the note stop need to be give the same program as the note start
                note_programs = events_programs[note_start_ids]
                note_channels = events.channel[note_start_ids]
                note_start_events = events[note_start_ids]
                note_stop_events = events[note_stop_ids]
                note_starts_time = events_times[note_start_ids]
                note_stops_time = events_times[note_stop_ids]

                note_groups = group_data([note_programs, note_channels])

        control_change_events_ids = np.nonzero(events.event_type == 3)[0]

        control_change_events = events[control_change_events_ids]
        channels_controls = {channel: ControlArray.zeros(0) for channel in range(16)}
        if len(control_change_events) > 0:
            channels_control_change_events_ids = group_data([control_change_events.channel], control_change_events_ids)

            for channel, channel_control_change_events_ids in channels_control_change_events_ids.items():
                channel_control_change_events = events[channel_control_change_events_ids]
                controls = ControlArray.zeros(len(channel_control_change_events_ids))
                controls.time = events_times[channel_control_change_events_ids]
                controls.tick = events_ticks[channel_control_change_events_ids]
                controls.number = channel_control_change_events.value1
                controls.value = channel_control_change_events.value2
                channels_controls[channel] = controls

        channels_pedals = {}
        for channel, channel_controls in channels_controls.items():
            if len(channel_controls) > 0:
                pedals = get_pedals_from_controls(channel_controls)
            else:
                pedals = PedalArray.zeros(0)
            channels_pedals[channel] = pedals

        assert len(events_groups) > 0
        for group_keys, track_events_ids in events_groups.items():
            if group_keys not in note_groups:
                track_notes_ids = np.zeros((0,), dtype=np.int32)
            else:
                track_notes_ids = note_groups[group_keys]
                track_notes_ids = np.sort(track_notes_ids)  # type: ignore # to keep the original order of the notes in the midi
            track_program, track_channel = group_keys
            assert track_program >= 0 and track_program < 128, "program should be between 0 and 127"
            assert track_channel >= 0 and track_channel < 16, "channel should be between 0 and 15"
            track_events = events[track_events_ids]
            track_events_times = events_times[track_events_ids]
            track_events_ticks = events_ticks[track_events_ids]
            assert np.all(np.diff(track_events_ticks) >= 0)

            pitch_bends_mask = track_events.event_type == 2
            pitch_bends_events = track_events[pitch_bends_mask]
            pitch_bends = PitchBendArray.zeros(len(pitch_bends_events))
            pitch_bends.time = track_events_times[pitch_bends_mask]
            pitch_bends.tick = track_events_ticks[pitch_bends_mask]
            pitch_bends.value = pitch_bends_events.value1

            # extract the event of type note on or note off
            if len(track_notes_ids) == 0:
                notes_np = NoteArray.zeros(0)
            else:
                notes_np = NoteArray.zeros(len(track_notes_ids))
                track_note_start_events = note_start_events[track_notes_ids]
                track_note_stop_events = note_stop_events[track_notes_ids]
                notes_np.start = note_starts_time[track_notes_ids]
                notes_np.start_tick = track_note_start_events.tick
                notes_np.duration = note_stops_time[track_notes_ids] - note_starts_time[track_notes_ids]
                notes_np.duration_tick = track_note_stop_events.tick.astype(
                    np.uint32
                ) - track_note_start_events.tick.astype(np.uint32)
                # assert np.all(notes_np.duration_tick > 0), "duration_tick should be strictly positive"
                notes_np.pitch = track_note_start_events.value1
                notes_np.velocity = track_note_start_events.value2

                # reorder using the original midi order
                notes_np = notes_np[np.argsort(note_start_ids[track_notes_ids])]

            track = Track(
                channel=int(track_channel),
                midi_track_id=midi_track_id,
                program=int(track_program),
                is_drum=track_channel == 9,  # FIXME
                name=midi_track.name,
                notes=notes_np,
                controls=channels_controls[track_channel],
                pedals=channels_pedals[track_channel],
                pitch_bends=pitch_bends,
            )
            tracks.append(track)

    return Score(
        tracks=tracks,
        lyrics=lyrics,
        last_tick=duration_tick,
        time_signature=signature,
        ticks_per_quarter=ticks_per_quarter,
        tempo=tempo,
    )


def has_duplicate_values(values: list) -> bool:
    list_values = list(values)
    return len(list_values) != len(set(list_values))


def score_to_midi(score: Score) -> Midi:
    """Convert a Score to a Midi."""
    midi_tracks = []

    use_multiple_tracks = len(set(track.midi_track_id for track in score.tracks)) > 1
    # use multiple track if more than 16 tracks
    if len(score.tracks) >= 16:
        use_multiple_tracks = True
    if has_duplicate_values([track.channel for track in score.tracks]):
        # multiple tracks with the same channel not supported because
        # it requires carefull program changes each time the instrument changes in the channel
        # using multiple tracks instead
        use_multiple_tracks = True

    if not use_multiple_tracks:
        num_events = 0
        for track in score.tracks:
            num_events += len(track.notes) * 2 + len(track.controls) + len(track.pitch_bends) + 1
        num_events += len(score.tempo)
        num_events += len(score.tracks)  # end of tracks event.
        events = EventArray.zeros(num_events)

        id_start = 0

        tempo = score.tempo
        events.tick[id_start : id_start + len(tempo)] = tempo.tick
        events.event_type[id_start : id_start + len(tempo)] = 5
        events.channel[id_start : id_start + len(tempo)] = 0
        events.value1[id_start : id_start + len(tempo)] = 60000000 / tempo.quarter_notes_per_minute
        events.value2[id_start : id_start + len(tempo)] = 0
        id_start += len(tempo)

        lyrics = score.lyrics
    else:
        # create track with the tempo changes and time signature
        tempo_events = EventArray.zeros(len(score.tempo))
        tempo_events.event_type[:] = 5
        tempo_events.channel[:] = 0
        tempo_events.value1[:] = 60000000 / score.tempo.quarter_notes_per_minute
        tempo_events.value2[:] = 0
        tempo_events.tick[:] = score.tempo.tick

        # add the time signature event
        signature_events = EventArray.zeros(len(score.time_signature))
        signature_events.event_type[:] = 10
        signature_events.channel[:] = 0
        signature_events.value1[:] = score.time_signature.numerator
        signature_events.value2[:] = np.log2(score.time_signature.denominator)
        signature_events.value3[:] = score.time_signature.clocks_per_click
        signature_events.value4[:] = score.time_signature.notated_32nd_notes_per_beat
        signature_events.tick[:] = score.time_signature.tick

        events = EventArray.concatenate([tempo_events, signature_events])
        midi_tracks.append(
            MidiTrack(
                name="tempo",
                lyrics=[],
                events=events,
            )
        )
        # create a track for the lyrics
        if score.lyrics is not None:
            midi_tracks.append(
                MidiTrack(
                    name="lyrics",
                    events=EventArray.zeros(0),
                    lyrics=score.lyrics,
                )
            )
        lyrics = None

    channels = attribute_midi_channels(score, max_channels=0, use_multiple_tracks=use_multiple_tracks)
    for track_id, track in enumerate(score.tracks):
        if use_multiple_tracks:
            num_events = len(track.notes) * 2 + len(track.controls) + len(track.pitch_bends) + 2
            events = EventArray.zeros(num_events)
            id_start = 0

        num_track_events = len(track.notes) * 2 + len(track.controls) + len(track.pitch_bends) + 2

        events.channel[id_start : id_start + num_track_events] = channels[track_id]

        # add the program change event
        events.event_type[id_start] = 4
        events.value1[id_start] = track.program
        events.value2[id_start] = 0
        events.tick[id_start] = 0
        id_start += 1

        # add the notes on events
        events.event_type[id_start : id_start + len(track.notes)] = 0
        events.value1[id_start : id_start + len(track.notes)] = track.notes.pitch
        events.value2[id_start : id_start + len(track.notes)] = track.notes.velocity
        events.tick[id_start : id_start + len(track.notes)] = track.notes.start_tick
        id_start += len(track.notes)

        # add the notes off events
        events.event_type[id_start : id_start + len(track.notes)] = 1
        events.value1[id_start : id_start + len(track.notes)] = track.notes.pitch
        events.value2[id_start : id_start + len(track.notes)] = 0
        events.tick[id_start : id_start + len(track.notes)] = track.notes.start_tick + track.notes.duration_tick
        # assert np.all(track.notes.duration_tick > 0), "duration_tick should be strictly positive"
        id_start += len(track.notes)

        # add the control change events
        events.event_type[id_start : id_start + len(track.controls)] = 3
        events.value1[id_start : id_start + len(track.controls)] = track.controls.number
        events.value2[id_start : id_start + len(track.controls)] = track.controls.value
        events.tick[id_start : id_start + len(track.controls)] = track.controls.tick
        id_start += len(track.controls)

        # TODO check that the pedals are consistent with the controls

        # add the pitch bend events
        events.event_type[id_start : id_start + len(track.pitch_bends)] = 2
        events.value1[id_start : id_start + len(track.pitch_bends)] = track.pitch_bends.value
        events.value2[id_start : id_start + len(track.pitch_bends)] = 0
        events.tick[id_start : id_start + len(track.pitch_bends)] = track.pitch_bends.tick
        id_start += len(track.pitch_bends)

        # add a end of track event based on the score duration

        events.event_type[id_start] = 9
        events.channel[id_start] = 0
        events.value1[id_start] = 0
        events.value2[id_start] = 0
        events.tick[id_start] = score.last_tick
        id_start += 1

        if use_multiple_tracks:
            order = np.lexsort((np.arange(len(events)), events.tick))
            events = events[order]
            midi_track = MidiTrack(
                name=track.name,
                events=events,
                lyrics=lyrics,
            )
            midi_tracks.append(midi_track)

    if not use_multiple_tracks:
        # sort by tick and keep the original order of the events
        # for event with the same tick,
        order = np.lexsort((np.arange(len(events)), events.tick))
        events = events[order]

        midi_track = MidiTrack(
            name=track.name,
            events=events,
            lyrics=lyrics,
        )
        midi_tracks = [midi_track]
    midi_score = Midi(tracks=midi_tracks, ticks_per_quarter=score.ticks_per_quarter)
    return midi_score


def load_score(
    file_path: str | Path,
    notes_mode: NotesMode = "note_off_stops_all",
    minimize_tempo: bool = True,
    check_round_trip: bool = False,
) -> Score:
    """Loads a MIDI file and converts it to a Score."""
    with open(file_path, "rb") as file:
        data = file.read()
    score = load_score_bytes(
        data, notes_mode=notes_mode, minimize_tempo=minimize_tempo, check_round_trip=check_round_trip
    )
    return score


def load_score_bytes(
    data: bytes,
    notes_mode: NotesMode = "note_off_stops_all",
    minimize_tempo: bool = True,
    check_round_trip: bool = False,
) -> Score:
    midi_raw = load_midi_bytes(data)
    score = midi_to_score(midi_raw, minimize_tempo=minimize_tempo, notes_mode=notes_mode)

    if check_round_trip:
        # check if the two scores can be converted back and forth
        midi_raw2 = score_to_midi(score)
        score2 = midi_to_score(midi_raw2, minimize_tempo=minimize_tempo, notes_mode=notes_mode)
        assert_scores_equal(score, score2)

    return score


def save_score_to_midi(score: Score, file_path: str, check_round_trip: bool = True) -> None:
    """Saves a Score to a MIDI file."""
    midi_score = score_to_midi(score)
    if check_round_trip:
        score2 = midi_to_score(midi_score, minimize_tempo=False, notes_mode="first_in_first_out")
        # We do not compare channels because they can be created in score_to_midi
        assert_scores_equal(score, score2, compare_channels=False)
    save_midi_file(midi_score, file_path)


def merge_non_overlapping_tracks(score: Score) -> Score:
    """Merge non overlapping tracks with same program into one track when possible.

    This is useful to reduce the number of tracks in the score when using
    MIDI soundfont synthetizers limited to 16 channels for example.
    """
    new_tracks: list[Track] = []
    for track in score.tracks:
        merged = False
        for new_track in new_tracks:
            if new_track.program == track.program and new_track.is_drum == track.is_drum:
                # tick of control does not matter before the first note
                first_note_tick = min(
                    np.min(new_track.notes.start_tick),
                    np.min(track.notes.start_tick),
                )
                # check controls pedals and pitch_bends are identical if not continue
                if not len(new_track.controls) == len(track.controls):
                    continue
                if not np.allclose(
                    np.clip(new_track.controls.tick, first_note_tick, None),
                    np.clip(track.controls.tick, first_note_tick, None),
                ):
                    continue
                if not np.array_equal(new_track.controls.value, track.controls.value):
                    continue
                if not np.array_equal(new_track.pedals._data, track.pedals._data):
                    continue
                if not np.array_equal(new_track.pitch_bends._data, track.pitch_bends._data):
                    continue

                # check if any note overlaps with the new track
                combined_notes = NoteArray.concatenate((new_track.notes, track.notes))
                overlapping_notes = get_overlapping_notes(combined_notes)
                if len(overlapping_notes) > 0:
                    # there are overlapping notes, skip the merge
                    continue

                # we can merge the tracks without loss
                new_track.notes = NoteArray.concatenate((new_track.notes, track.notes))

                merged = True
                break
        if not merged:
            # add the track to the new tracks
            new_tracks.append(deepcopy(track))

    new_score = Score(
        tracks=new_tracks,
        last_tick=score.last_tick,
        time_signature=score.time_signature,
        ticks_per_quarter=score.ticks_per_quarter,
        tempo=score.tempo,
        lyrics=score.lyrics,
    )
    return new_score


def attribute_midi_channels(score: Score, max_channels: int, use_multiple_tracks: bool) -> dict[int, int]:
    """Attribute channels to tracks with constrain on the drum tracks."""
    if max_channels == 0:
        max_channels = max(16, len(score.tracks) + 1)

    if use_multiple_tracks:
        channel_mapping = {}
        for track_id, track in enumerate(score.tracks):
            if track.is_drum:
                channel_mapping[track_id] = DRUM_CHANNEL
            elif track.channel and track.channel < max_channels:
                channel_mapping[track_id] = track.channel
            else:
                channel_mapping[track_id] = 0
    else:
        if len(score.tracks) > max_channels:
            raise ValueError("MIDI only supports 16 channels.")

        available_channels = [i for i in range(max_channels)]
        # the channel 9 is reserved for drums
        available_channels.remove(DRUM_CHANNEL)

        dum_channel_used = False
        channel_mapping = {}

        for track_id, track in enumerate(score.tracks):
            if track.is_drum:
                channel_mapping[track_id] = DRUM_CHANNEL
                if dum_channel_used:
                    raise ValueError("Drum channel already used")
                dum_channel_used = True
            else:
                if len(available_channels) == 0:
                    raise ValueError("No available channels left")
                if track.channel in available_channels:
                    available_channels.remove(track.channel)
                    channel_mapping[track_id] = track.channel
                else:
                    channel_mapping[track_id] = available_channels.pop(0)
    return channel_mapping


def filter_instruments(score: Score, instrument_names: list[str]) -> Score:
    """Filter the tracks of the score to keep only the ones with the specified instrument names."""
    tracks = []

    programs = set([instrument_to_program[instrument_name] for instrument_name in instrument_names])
    for track in score.tracks:
        if track.is_drum:
            continue
        if track.program in programs:
            tracks.append(track)
    return Score(
        tracks=tracks,
        last_tick=score.last_tick,
        time_signature=score.time_signature,
        ticks_per_quarter=score.ticks_per_quarter,
        tempo=score.tempo,
        lyrics=score.lyrics,
    )


def remove_empty_tracks(score: Score) -> Score:
    """Remove the tracks of the score that have no notes."""
    tracks = []
    for track in score.tracks:
        if track.notes.size > 0:
            tracks.append(track)
    return Score(
        tracks=tracks,
        last_tick=score.last_tick,
        time_signature=score.time_signature,
        ticks_per_quarter=score.ticks_per_quarter,
        tempo=score.tempo,
        lyrics=score.lyrics,
    )


def remove_pitch_bends(score: Score) -> Score:
    """Remove the pitch bends from the score."""
    tracks = []
    for track in score.tracks:
        new_track = copy(track)
        new_track.pitch_bends = PitchBendArray.zeros(0)
        tracks.append(new_track)

    return Score(
        tracks=tracks,
        last_tick=score.last_tick,
        time_signature=score.time_signature,
        ticks_per_quarter=score.ticks_per_quarter,
        tempo=score.tempo,
        lyrics=score.lyrics,
    )


def remove_pedals(score: Score) -> Score:
    """Remove the pedals from the score."""
    tracks = []
    for track in score.tracks:
        new_track = copy(track)
        new_track.pedals = PedalArray.zeros(0)
        tracks.append(new_track)

    return Score(
        tracks=tracks,
        last_tick=score.last_tick,
        time_signature=score.time_signature,
        ticks_per_quarter=score.ticks_per_quarter,
        tempo=score.tempo,
        lyrics=score.lyrics,
    )


def remove_control_changes(score: Score) -> Score:
    """Remove the control changes from the score."""
    tracks = []
    for track in score.tracks:
        new_track = copy(track)
        new_track.controls = ControlArray.zeros(0)
        tracks.append(new_track)

    return Score(
        tracks=tracks,
        last_tick=score.last_tick,
        time_signature=score.time_signature,
        ticks_per_quarter=score.ticks_per_quarter,
        tempo=score.tempo,
        lyrics=score.lyrics,
    )


def filter_pitch(score: Score, pitch_min: int, pitch_max: int) -> Score:
    tracks = []
    for track in score.tracks:
        new_track = copy(track)
        keep = (track.notes.pitch >= pitch_min) & (track.notes.pitch < pitch_max)
        new_track.notes = track.notes[keep]
        tracks.append(new_track)
    return Score(
        tracks=tracks,
        last_tick=score.last_tick,
        time_signature=score.time_signature,
        ticks_per_quarter=score.ticks_per_quarter,
        tempo=score.tempo,
        lyrics=score.lyrics,
    )


def get_overlapping_notes(notes: NoteArray) -> np.ndarray:
    order = np.lexsort((notes.start, notes.pitch))
    overlapping_notes = _get_overlapping_notes_pairs_jit(notes.start, notes.duration, notes.pitch, order)
    return overlapping_notes


def get_overlapping_notes_ticks(notes: NoteArray) -> np.ndarray:
    order = np.lexsort((notes.start_tick, notes.pitch))
    overlapping_notes = _get_overlapping_notes_pairs_jit(notes.start_tick, notes.duration_tick, notes.pitch, order)
    return overlapping_notes


def check_no_overlapping_notes(notes: NoteArray, use_ticks: bool = True) -> None:
    """Check that there are no overlapping notes at the same pitch."""
    if use_ticks:
        overlapping_notes = get_overlapping_notes_ticks(notes)
    else:
        overlapping_notes = get_overlapping_notes(notes)
    if len(overlapping_notes) > 0:
        raise ValueError("Overlapping notes found")


def check_no_overlapping_notes_in_score(score: Score) -> None:
    for track in score.tracks:
        check_no_overlapping_notes(track.notes)


def time_to_float_tick(time: float, tempo: TempoArray, ticks_per_quarter: int) -> float:
    """Convert a time in seconds to tick."""
    # get the tempo at the start of the time range
    tempo_idx: int = 0
    if tempo.size > 1:
        tempo_idx = int(np.searchsorted(tempo.time, time, side="right") - 1)

    tempo_idx = np.clip(tempo_idx, 0, tempo.size - 1)
    quarter_notes_per_minute = tempo.quarter_notes_per_minute[tempo_idx]
    ref_ticks = tempo.tick[tempo_idx]
    ref_time = tempo.time[tempo_idx]
    quarter_per_second = quarter_notes_per_minute / 60.0
    ticks_per_second = ticks_per_quarter * quarter_per_second
    return ref_ticks + (time - ref_time) * ticks_per_second


def round_to_tick(time: float, tempo: TempoArray, ticks_per_quarter: int) -> tuple[float, int]:
    """Convert a time in seconds to tick and round it to the nearest tick."""
    tempo_idx: int = 0
    if tempo.size > 1:
        tempo_idx = int(np.searchsorted(tempo.time, time, side="right") - 1)

    tempo_idx = np.clip(tempo_idx, 0, tempo.size - 1)
    quarter_notes_per_minute = tempo.quarter_notes_per_minute[tempo_idx]
    ref_ticks = tempo.tick[tempo_idx]
    ref_time = tempo.time[tempo_idx]
    quarter_per_second = quarter_notes_per_minute / 60.0
    ticks_per_second = ticks_per_quarter * quarter_per_second
    tick_float = ref_ticks + (time - ref_time) * ticks_per_second
    tick = int(np.round(tick_float))
    time_rounded = ref_time + (tick - ref_ticks) / ticks_per_second
    return time_rounded, tick


def round_to_ticks(time: np.ndarray, tempo: TempoArray, ticks_per_quarter: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a time in seconds to tick and round it to the nearest tick."""
    # get the tempo at the start of the time range
    if tempo.size > 1:
        tempo_idx = np.searchsorted(tempo.time, time, side="right") - 1
    else:
        tempo_idx = np.zeros(len(time), dtype=np.int32)
    tempo_idx = np.clip(tempo_idx, 0, tempo.size - 1)
    quarter_notes_per_minute = tempo.quarter_notes_per_minute[tempo_idx]
    ref_ticks = tempo.tick[tempo_idx]
    ref_time = tempo.time[tempo_idx]
    quarter_per_second = quarter_notes_per_minute / 60.0
    ticks_per_second = ticks_per_quarter * quarter_per_second
    tick_float = ref_ticks + (time - ref_time) * ticks_per_second
    tick = np.round(tick_float).astype(np.int32)
    time_rounded = ref_time + (tick - ref_ticks) / ticks_per_second
    return time_rounded, tick


def time_to_tick(time: float, tempo: TempoArray, ticks_per_quarter: int) -> int:
    """Convert a time in seconds to tick."""
    return int(np.round(time_to_float_tick(time, tempo, ticks_per_quarter)))


def tick_to_time(tick: float, tempo: TempoArray, ticks_per_quarter: int) -> float:
    """Convert a tick to time in seconds."""
    # get the tempo at the start of the time range
    tempo_idx = np.searchsorted(tempo.tick, tick, side="right") - 1
    tempo_idx = np.clip(tempo_idx, 0, tempo.size - 1)
    quarter_notes_per_minute = tempo.quarter_notes_per_minute[tempo_idx]
    ref_ticks = tempo.tick[tempo_idx]
    ref_time = tempo.time[tempo_idx]
    quarter_per_second = quarter_notes_per_minute / 60.0
    ticks_per_second = ticks_per_quarter * quarter_per_second
    return ref_time + (tick - ref_ticks) / ticks_per_second


def ticks_to_times(tick: np.ndarray, tempo: TempoArray, ticks_per_quarter: int) -> np.ndarray:
    """Convert a tick to time in seconds."""
    # get the tempo at the start of the time range
    tempo_idx = np.searchsorted(tempo.tick, tick, side="right") - 1
    tempo_idx = np.clip(tempo_idx, 0, tempo.size - 1)
    quarter_notes_per_minute = tempo.quarter_notes_per_minute[tempo_idx]
    ref_ticks = tempo.tick[tempo_idx]
    ref_time = tempo.time[tempo_idx]
    quarter_per_second = quarter_notes_per_minute / 60.0
    ticks_per_second = ticks_per_quarter * quarter_per_second
    return ref_time + (tick - ref_ticks) / ticks_per_second


def times_to_float_ticks(time: np.ndarray, tempo: TempoArray, ticks_per_quarter: int) -> np.ndarray:
    """Convert a time in seconds to ticks."""
    # get the tempo at the start of the time range
    if tempo.size > 1:
        tempo_idx = np.searchsorted(tempo.time, time, side="right") - 1
    else:
        tempo_idx = np.zeros(len(time), dtype=np.int32)
    tempo_idx = np.clip(tempo_idx, 0, tempo.size - 1)
    quarter_notes_per_minute = tempo.quarter_notes_per_minute[tempo_idx]
    ref_ticks = tempo.tick[tempo_idx]
    ref_time = tempo.time[tempo_idx]
    quarter_per_second = quarter_notes_per_minute / 60.0
    ticks_per_second = ticks_per_quarter * quarter_per_second
    return ref_ticks + (time - ref_time) * ticks_per_second


def times_to_ticks(time: np.ndarray, tempo: TempoArray, ticks_per_quarter: int) -> np.ndarray:
    """Convert a time in seconds to ticks."""
    return np.round(times_to_float_ticks(time, tempo, ticks_per_quarter)).astype(np.int32)


def update_ticks(score: Score, tempo: TempoArray) -> Score:
    """Update the ticks of the score according to the tempo."""
    tracks = []
    for track in score.tracks:
        new_track = copy(track)
        new_track.notes.start_tick = times_to_ticks(new_track.notes.start, tempo, score.ticks_per_quarter)
        new_track.notes.duration_tick = (
            times_to_ticks(new_track.notes.start + new_track.notes.duration, tempo, score.ticks_per_quarter)
            - new_track.notes.start_tick
        )
        new_track.pedals.tick = times_to_ticks(new_track.pedals.time, tempo, score.ticks_per_quarter)
        new_track.controls.tick = times_to_ticks(new_track.controls.time, tempo, score.ticks_per_quarter)
        new_track.pitch_bends.tick = times_to_ticks(new_track.pitch_bends.time, tempo, score.ticks_per_quarter)
        tracks.append(new_track)
    last_tick = int(np.ceil(time_to_float_tick(score.last_tick, tempo, score.ticks_per_quarter)))

    return Score(
        tracks=tracks,
        last_tick=last_tick,
        time_signature=score.time_signature,
        ticks_per_quarter=score.ticks_per_quarter,
        tempo=tempo,
    )


def crop_score(score: Score, start: float, end: float) -> Score:
    """Crop a MIDI score to a specific time range.

    Note: the NoteOn events from before the start time are not kept
    and thus the sound may not be the same as the cropped original sound.
    """
    tracks = []

    previous_tempos = np.nonzero(score.tempo.time < start)[0]
    tempo_keep = (score.tempo.time < end) & (score.tempo.time >= start)
    if len(previous_tempos) > 0:
        tempo_keep[previous_tempos[-1]] = True
    new_tempo = score.tempo[tempo_keep]
    new_tempo.time = np.maximum(new_tempo.time - start, 0)
    tick_end = int(np.floor(time_to_float_tick(end, score.tempo, score.ticks_per_quarter)))
    tick_start = int(np.ceil(time_to_float_tick(start, score.tempo, score.ticks_per_quarter)))
    new_tempo.tick = np.maximum(new_tempo.tick - tick_start, 0)

    for track in score.tracks:
        notes = track.notes
        notes_end = notes.start + notes.duration
        notes_end_tick = notes.start_tick + notes.duration_tick
        notes_keep = (notes.start < end) & (notes_end > start)
        new_notes = notes[notes_keep]
        new_notes.start = np.maximum(new_notes.start - start, 0)
        new_notes_end = np.minimum(notes_end[notes_keep] - start, end - start)
        new_notes.duration = new_notes_end - new_notes.start
        new_notes.start_tick = np.maximum(new_notes.start_tick - tick_start, 0)
        new_notes_end_tick = np.minimum(notes_end_tick[notes_keep] - tick_start, tick_end - tick_start)
        new_notes.duration_tick = new_notes_end_tick - new_notes.start_tick

        assert np.all(new_notes_end <= end - start), "Note end time exceeds score duration"
        # remove note with duration 0
        new_notes = new_notes[new_notes.duration_tick > 0]

        pedals_end = track.pedals.time + track.pedals.duration
        pedals_keep = (track.pedals.time < end) & (pedals_end > start)
        # keep the last pedal before the start time
        last_previous_pedal = np.nonzero(track.pedals.time < start)[0]
        pedals_keep[last_previous_pedal] = True
        new_pedals = track.pedals[pedals_keep]
        new_pedals_end = np.minimum(pedals_end[pedals_keep], end) - start
        new_pedals.duration = new_pedals_end - new_pedals.time
        new_pedals.time = np.maximum(new_pedals.time - start, 0)
        new_pedals.tick = np.maximum(new_pedals.tick - tick_start, 0)

        controls_keep = (track.controls.time < end) & (track.controls.time >= start)
        # for each control number keep the last value before the start time
        previous_controls = track.controls[track.controls.time < start]
        last_control = np.full((127,), -1, dtype=np.int32)
        last_control[previous_controls.number] = previous_controls.value
        controls_keep[last_control[last_control >= 0]] = True
        new_controls = track.controls[controls_keep]
        new_controls.time = np.maximum(new_controls.time - start, 0)
        new_controls.tick = np.maximum(new_controls.tick - tick_start, 0)

        pitch_bends_keep = (track.pitch_bends.time < end) & (track.pitch_bends.time >= start)
        # keep the last pitch bend before the start time
        last_previous_pitch_bend = np.nonzero(track.pitch_bends.time < start)[0]
        pitch_bends_keep[last_previous_pitch_bend] = True
        new_pitch_bends = track.pitch_bends[pitch_bends_keep]
        new_pitch_bends.time = np.maximum(new_pitch_bends.time - start, 0)
        new_pitch_bends.tick = np.maximum(new_pitch_bends.tick - tick_start, 0)

        new_track = Track(
            channel=track.channel,
            program=track.program,
            is_drum=track.is_drum,
            name=track.name,
            notes=new_notes,
            controls=new_controls,
            pedals=new_pedals,
            pitch_bends=new_pitch_bends,
            midi_track_id=track.midi_track_id,
        )
        tracks.append(new_track)
    return Score(
        tracks=tracks,
        last_tick=tick_end - tick_start,
        time_signature=score.time_signature,
        ticks_per_quarter=score.ticks_per_quarter,
        tempo=new_tempo,
    )


def select_tracks(score: Score, track_ids: list[int]) -> Score:
    """Select only the tracks with the specified programs."""
    tracks = [score.tracks[track_id] for track_id in track_ids]
    return Score(
        tracks=tracks,
        last_tick=score.last_tick,
        time_signature=score.time_signature,
        ticks_per_quarter=score.ticks_per_quarter,
        tempo=score.tempo,
        lyrics=score.lyrics,
    )


def distance(score1: Score, score2: Score, sort_tracks_with_programs: bool = False) -> float:
    assert len(score1.tracks) == len(score2.tracks), "The scores have different number of tracks"
    max_diff = 0
    tracks_1 = score1.tracks
    tracks_2 = score2.tracks
    if sort_tracks_with_programs:
        tracks_1 = sorted(tracks_1, key=lambda x: x.program)
        tracks_2 = sorted(tracks_2, key=lambda x: x.program)
    for track1, track2 in zip(tracks_1, tracks_2, strict=False):
        print("Programs", track1.program, track2.program)
        # print("Notes", track1.notes.shape, track2.notes.shape)
        print("Controls", track1.controls.size, track2.controls.size)
        print("Pedals", track1.pedals.size, track2.pedals.size)
        print("Pitch bends", track1.pitch_bends.size, track2.pitch_bends.size)

        # max note time difference
        max_start_time_diff = max(abs(track1.notes.start - track2.notes.start))
        max_duration_diff = max(abs(track1.notes.duration - track2.notes.duration))
        print("Max note start time difference", max_start_time_diff)
        print("Max note duration difference", max_duration_diff)
        max_diff = max(max_diff, max_start_time_diff, max_duration_diff)
        # max control time difference
        max_control_time_diff = max(abs(track1.controls.time - track2.controls.time))
        print("Max control time difference", max_control_time_diff)
        max_diff = max(max_diff, max_control_time_diff)
        # max pedal time difference
        if track1.pedals.size > 0:
            max_pedal_time_diff = max(abs(track1.pedals.time - track2.pedals.time))
            print("Max pedal time difference", max_pedal_time_diff)
            max_diff = max(max_diff, max_pedal_time_diff)
        # max pitch bend time difference
        if track1.pitch_bends.size > 0:
            max_pitch_bend_time_diff = max(abs(track1.pitch_bends.time - track2.pitch_bends.time))
            print("Max pitch bend time difference", max_pitch_bend_time_diff)
            max_diff = max(max_diff, max_pitch_bend_time_diff)
    return max_diff


def assert_scores_equal(
    score1: Score,
    score2: Score,
    time_tol: float = 1e-3,
    value_tol: float = 1e-2,
    tick_tol: int = 0,
    compare_channels: bool = True,
) -> None:
    assert len(score1.tracks) == len(score2.tracks), "The scores have different number of tracks"
    max_diff = 0
    tracks_1 = score1.tracks
    tracks_2 = score2.tracks

    # sort by nme, program and then by number of notes to try to have the same order
    if compare_channels:
        tracks_1 = sorted(tracks_1, key=lambda x: (x.name, x.program, x.channel, len(x.notes), x.notes.pitch.sum()))
        tracks_2 = sorted(tracks_2, key=lambda x: (x.name, x.program, x.channel, len(x.notes), x.notes.pitch.sum()))
    else:
        tracks_1 = sorted(tracks_1, key=lambda x: (x.name, x.program, len(x.notes), x.notes.pitch.sum()))
        tracks_2 = sorted(tracks_2, key=lambda x: (x.name, x.program, len(x.notes), x.notes.pitch.sum()))
    assert score1.tempo.size == score2.tempo.size, "Different number of tempo events"
    assert np.all(score1.tempo.tick == score2.tempo.tick), "Different tick values for tempo events"

    assert np.allclose(score1.tempo.quarter_notes_per_minute, score2.tempo.quarter_notes_per_minute, atol=1e-3), (
        "Different quarter_notes_per_minute values for tempo events"
    )
    assert np.allclose(score1.tempo.time, score2.tempo.time, atol=1e-3), "Different time values for tempo events"
    for track_id, (track1, track2) in enumerate(zip(tracks_1, tracks_2, strict=False)):
        assert track1.name == track2.name, "Track names are different"
        assert track1.program == track2.program, "Track programs are different"
        if compare_channels:
            assert track1.channel == track2.channel, "Track channels are different"
        # sort not by pitch then tick
        # notes1= track1.notes
        # notes2= track2.notes
        order1 = np.lexsort((np.arange(len(track1.notes)), track1.notes.start_tick, track1.notes.pitch))
        notes1 = track1.notes[order1]
        order2 = np.lexsort((np.arange(len(track2.notes)), track2.notes.start_tick, track2.notes.pitch))
        notes2 = track2.notes[order2]

        min_len = min(len(notes1), len(notes2))
        np.nonzero(notes1[:min_len].start_tick != notes2[:min_len].start_tick)
        assert len(notes1) == len(notes2), f"Different number of notes in track {track_id}"
        if len(notes1) > 0:
            assert np.all(notes1.pitch == notes2.pitch), f"Pitches are different in track {track_id}"
            max_tick_diff = max(abs(notes1.start_tick - notes2.start_tick))
            assert max_tick_diff <= tick_tol, f"Tick difference larger than {tick_tol} in track {track_id}"
            max_duration_tick_diff = max(abs(notes1.duration_tick - notes2.duration_tick))
            assert max_duration_tick_diff <= tick_tol, (
                f"Duration tick difference {max_duration_tick_diff} greater than {tick_tol} in track {track_id}"
            )
            # max note time difference
            max_start_time_diff = max(abs(notes1.start - notes2.start))
            assert max_start_time_diff <= time_tol, (
                f"Max note start time difference {max_start_time_diff}>{time_tol} in track {track_id}"
            )
            notes_stop_1 = notes1.start + notes1.duration
            notes_stop_2 = notes2.start + notes2.duration
            max_stop_diff = max(abs(notes_stop_1 - notes_stop_2))
            assert max_stop_diff <= time_tol, f"Max note end difference {max_stop_diff}>{time_tol} in track {track_id}"
            # max note velocity difference
            velocify_abs_diff = abs(notes1.velocity.astype(np.int16) - notes2.velocity.astype(np.int16))
            max_velocity_diff = max(velocify_abs_diff)
            assert max_velocity_diff <= value_tol, (
                f"Max note velocity difference {max_velocity_diff}>{value_tol} in track {track_id}"
            )
            # max note duration difference
            max_duration_diff = max(abs(notes1.duration - notes2.duration))
            assert max_duration_diff <= time_tol, (
                f"Max note duration difference {max_duration_diff}>{time_tol} in track {track_id}"
            )

        # max control time difference
        assert track1.controls.size == track2.controls.size, f"Different number of control events in track {track_id}"
        if track1.controls.size > 0:
            max_control_time_diff = max(abs(track1.controls.time - track2.controls.time))
            assert max_control_time_diff <= time_tol, (
                f"Max control time difference {max_control_time_diff}>{time_tol} in track {track_id}"
            )
            max_diff = max(max_diff, max_control_time_diff)
        # max pedal time difference
        assert track1.pedals.size == track2.pedals.size, f"Different number of pedal events in track {track_id}"
        if track1.pedals.size > 0:
            max_pedal_time_diff = max(abs(track1.pedals.time - track2.pedals.time))
            max_diff = max(max_diff, max_pedal_time_diff)
            assert max_pedal_time_diff <= time_tol, (
                f"Max pedal time difference {max_pedal_time_diff}>{time_tol} in track {track_id}"
            )
        # max pitch bend time difference
        assert track1.pitch_bends.size == track2.pitch_bends.size, (
            f"Different number of pitch bend events in track {track_id}"
        )
        if track1.pitch_bends.size > 0:
            max_pitch_bend_time_diff = max(abs(track1.pitch_bends.time - track2.pitch_bends.time))
            max_diff = max(max_diff, max_pitch_bend_time_diff)
            assert max_pitch_bend_time_diff <= time_tol, (
                f"Max pitch bend time difference {max_pitch_bend_time_diff}>{time_tol} in track {track_id}"
            )


def get_score_instruments(score: Score) -> list[str]:
    """Get the instruments from a score."""
    instruments = set()
    for track in score.tracks:
        instrument_name = program_to_instrument[track.program]
        instruments.add(instrument_name)
    return list(instruments)


def get_score_instrument_groups(score: Score) -> list[str]:
    """Get the instrument groups from a score."""
    instrument_groups = set()
    for track in score.tracks:
        instrument_group_name = program_to_instrument_group[track.program]
        instrument_groups.add(instrument_group_name)
    return list(instrument_groups)


def get_num_notes_per_group(score: Score) -> dict[str, int]:
    """Get the number of notes per instrument group."""
    num_notes_per_group = {}
    for track in score.tracks:
        instrument_group_name = program_to_instrument_group[track.program]
        if instrument_group_name not in num_notes_per_group:
            num_notes_per_group[instrument_group_name] = 0
        num_notes_per_group[instrument_group_name] += len(track.notes)
    return num_notes_per_group
