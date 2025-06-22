"""Tools to process piano roll reprentationss of MIDI scores."""

from dataclasses import dataclass
from math import ceil
from typing import Optional

from numba.core.decorators import njit
import numpy as np

from numba_midi.score import (
    check_no_overlapping_notes,
    check_no_overlapping_notes_in_score,
    ControlArray,
    NoteArray,
    PedalArray,
    PitchBendArray,
    Score,
    SignatureArray,
    TempoArray,
    time_to_float_tick,
    times_to_ticks,
    Track,
)


@dataclass
class PianoRoll:
    """Dataclass for multi-track piano rolls."""

    array: np.ndarray
    time_step: float
    pitch_min: int
    num_bin_per_semitone: int
    programs: list[int]
    track_names: list[str]
    time_signature: SignatureArray
    ticks_per_quarter: int = 480
    midi_track_ids: Optional[list[int | None]] = None
    channels: Optional[list[int | None]] = None
    tempo: Optional[TempoArray] = None

    @property
    def duration(self) -> float:
        return self.array.shape[2] * self.time_step

    @property
    def pitch_max(self) -> int:
        return ceil(self.array.shape[1] / self.num_bin_per_semitone + self.pitch_min)

    @property
    def num_tracks(self) -> int:
        return self.array.shape[0]

    def __post_init__(self) -> None:
        assert len(self.array.shape) == 3, "Piano roll must be 3D (num_tracks, num_pitch, num_time)"
        assert self.array.dtype == np.uint8, "Piano roll must be uint8"
        assert self.array.shape[1] % self.num_bin_per_semitone == 0, (
            "Piano roll shape[0] must be divisible by num_bin_per_semitone"
        )
        assert self.array.shape[2] > 0, "Piano roll shape[1] must be greater than 0"
        assert self.pitch_min >= 0, "pitch_min must be greater than or equal to 0"
        if self.midi_track_ids is not None:
            assert len(self.midi_track_ids) == self.num_tracks, "midi_track_ids must be the same length as num_tracks"
        if self.channels is not None:
            assert len(self.channels) == self.num_tracks, "channels must be the same length as num_tracks"
        if self.track_names is not None:
            assert len(self.track_names) == self.num_tracks, "track_names must be the same length as num_tracks"

    @staticmethod
    def from_score(
        score: Score,
        time_step: float,
        pitch_min: int,
        pitch_max: int,
        num_bin_per_semitone: int,
        shorten_notes: bool = True,
        antialiasing: bool = False,
    ) -> "PianoRoll":
        """Create a PianoRoll from a Score."""
        return score_to_piano_roll(
            score=score,
            time_step=time_step,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            num_bin_per_semitone=num_bin_per_semitone,
            shorten_notes=shorten_notes,
            antialiasing=antialiasing,
        )

    def to_score(self, threshold: float = 0.0) -> Score:
        """Create a Score from a PianoRoll."""
        return piano_roll_to_score(
            piano_roll=self,
            threshold=threshold,
        )


@njit(cache=True)
def _add_notes_to_piano_roll_jit(
    piano_roll: np.ndarray,
    pitch: np.ndarray,
    start: np.ndarray,
    duration: np.ndarray,
    velocity: np.ndarray,
    time_step: float,
    pitch_min: int,
    pitch_max: int,
    num_bin_per_semitone: int,
    shorten_notes: bool,
    antialiasing: bool = False,
) -> None:
    # TODO take pitch bend into account
    for note_id in range(len(pitch)):
        note_pitch = pitch[note_id]
        note_start = start[note_id]
        note_duration = duration[note_id]
        note_velocity = velocity[note_id]

        assert note_pitch >= pitch_min and note_pitch < pitch_max, "Pitch out of range"
        col_start_float = note_start / time_step
        note_end = note_start + note_duration

        row = (note_pitch - pitch_min) * num_bin_per_semitone

        col_end_float = note_end / time_step
        if antialiasing:
            if shorten_notes:
                note_end = note_end - 2 * time_step
            alpha_start = 1.0 - (col_start_float - int(col_start_float))
            alpha_end = col_end_float - int(col_end_float)
            piano_roll[row, int(col_start_float)] += alpha_start * note_velocity
            if alpha_end > 0:
                piano_roll[row, int(col_end_float)] += alpha_end * note_velocity
            for col in range(int(col_start_float) + 1, int(col_end_float)):
                piano_roll[row, col] += note_velocity

        else:
            col_start_int = int(col_start_float)
            col_end_int = int(col_end_float)
            if shorten_notes:
                col_end_int -= 1
            # Note: short notes can be discarded if col_start_int==col_end_int
            for col in range(col_start_int, col_end_int):
                piano_roll[row, col] += note_velocity


def score_to_piano_roll(
    score: Score,
    time_step: float,
    pitch_min: int,
    pitch_max: int,
    num_bin_per_semitone: int,
    shorten_notes: bool = True,
    antialiasing: bool = False,
) -> PianoRoll:
    """Create a piano roll representation of the score.

    use shorten_notes to shorten the notes by 2*time_step to have gaps between notes
    """
    check_no_overlapping_notes_in_score(score)
    num_cols = int(np.ceil(score.duration / time_step))
    num_rows = (pitch_max - pitch_min) * num_bin_per_semitone
    num_tracks = len(score.tracks)
    piano_roll = np.zeros((num_tracks, num_rows, num_cols), dtype=float)

    for track_id, track in enumerate(score.tracks):
        _add_notes_to_piano_roll_jit(
            piano_roll=piano_roll[track_id],
            pitch=track.notes.pitch,
            start=track.notes.start,
            duration=track.notes.duration,
            velocity=track.notes.velocity,
            time_step=time_step,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            num_bin_per_semitone=num_bin_per_semitone,
            shorten_notes=shorten_notes,
            antialiasing=antialiasing,
        )

    return PianoRoll(
        array=piano_roll.astype(np.uint8),
        pitch_min=pitch_min,
        time_step=time_step,
        num_bin_per_semitone=num_bin_per_semitone,
        channels=[track.channel for track in score.tracks],
        programs=[track.program for track in score.tracks],
        midi_track_ids=[track.midi_track_id for track in score.tracks],
        time_signature=score.time_signature,
        ticks_per_quarter=score.ticks_per_quarter,
        tempo=score.tempo,
        track_names=[track.name for track in score.tracks],
    )


@njit(cache=True)
def _piano_roll_to_score_jit(
    piano_roll: np.ndarray, time_step: float, pitch_min: int = 0, threshold: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start_list = []
    duration_list = []
    pitch_list = []
    velocity_list = []

    for row in range(piano_roll.shape[0]):
        note_pitch = pitch_min + row
        note_on = False
        sum_velocity = 0
        col_start = 0
        for col in range(piano_roll.shape[1]):
            if piano_roll[row, col] > threshold:
                if not note_on:
                    col_start = col
                    note_on = True
                    sum_velocity = int(piano_roll[row, col])
                else:
                    sum_velocity += int(piano_roll[row, col])
            elif note_on:
                note_start = col_start * time_step
                duration_col = col - col_start
                note_duration = (col - col_start) * time_step
                note_velocity = np.uint8(sum_velocity / duration_col)
                note_on = False
                start_list.append(note_start)
                duration_list.append(note_duration)
                pitch_list.append(note_pitch)
                velocity_list.append(note_velocity)

        # deal with note still on on the last column
        if note_on:
            note_start = col_start * time_step
            col = piano_roll.shape[1]
            duration_col = col - col_start
            note_duration = (col - col_start) * time_step
            note_velocity = np.uint8(sum_velocity / duration_col)
            start_list.append(note_start)
            duration_list.append(note_duration)
            pitch_list.append(note_pitch)
            velocity_list.append(note_velocity)

    start = np.array(start_list, dtype=np.float32)
    duration = np.array(duration_list, dtype=np.float32)
    pitch = np.array(pitch_list, dtype=np.uint8)
    velocity = np.array(velocity_list, dtype=np.uint8)
    return start, duration, pitch, velocity


def piano_roll_to_score(
    piano_roll: PianoRoll,
    threshold: float = 0.0,
) -> Score:
    """Create a score from a piano roll representation."""
    assert piano_roll.array.dtype == np.uint8, "Piano roll array must be of type uint8"
    piano_roll_semitone = piano_roll.array.reshape(
        piano_roll.array.shape[0], -1, piano_roll.num_bin_per_semitone, piano_roll.array.shape[2]
    ).max(axis=2)
    tracks = []

    if piano_roll.tempo is None:
        # default tempo is 120 BPM
        tempo = TempoArray(time=[0], tick=[0], quarter_notes_per_minute=[120])
    else:
        tempo = piano_roll.tempo
    for track_id in range(piano_roll_semitone.shape[0]):
        start, duration, pitch, velocity = _piano_roll_to_score_jit(
            piano_roll=piano_roll_semitone[track_id],
            time_step=piano_roll.time_step,
            pitch_min=piano_roll.pitch_min,
            threshold=threshold,
        )
        # assert velocity.max() <= piano_roll_semitone[track_id].max()
        channel = piano_roll.channels[track_id] if piano_roll.channels is not None else None
        program = piano_roll.programs[track_id]
        midi_track_id = piano_roll.midi_track_ids[track_id] if piano_roll.midi_track_ids is not None else None

        # Create the structured array
        notes = NoteArray.zeros(len(start))

        # Assign values to the fields
        notes.start = start
        notes.duration = duration
        notes.pitch = pitch
        notes.velocity = velocity

        controls = ControlArray.zeros(0)
        pedals = PedalArray.zeros(0)
        pitch_bends = PitchBendArray.zeros(0)

        if piano_roll.ticks_per_quarter is None:
            ticks_per_quarter = 480
        else:
            ticks_per_quarter = piano_roll.ticks_per_quarter

        notes.start_tick = times_to_ticks(notes.start, tempo, ticks_per_quarter).astype(np.int32)

        notes_ends = notes.start + notes.duration
        notes_ends_tick = times_to_ticks(notes_ends, tempo, ticks_per_quarter).astype(np.int32)
        notes.duration_tick = notes_ends_tick - notes.start_tick
        check_no_overlapping_notes(notes)
        track = Track(
            program=program,
            is_drum=False,
            name=piano_roll.track_names[track_id],
            notes=notes,
            controls=controls,
            pedals=pedals,
            pitch_bends=pitch_bends,
            channel=channel,
            midi_track_id=midi_track_id,
        )
        tracks.append(track)
    last_tick = int(ceil(time_to_float_tick(piano_roll.duration, tempo, piano_roll.ticks_per_quarter)))
    return Score(
        tracks=tracks,
        last_tick=last_tick,
        time_signature=piano_roll.time_signature,
        tempo=tempo,
        ticks_per_quarter=piano_roll.ticks_per_quarter,
    )
