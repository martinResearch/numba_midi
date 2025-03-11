"""Tools to process piano roll reprentationss of MIDI scores."""

from dataclasses import dataclass
from math import ceil

from numba.core.decorators import njit
import numpy as np

from numba_midi.score import (
    check_no_overlapping_notes,
    check_no_overlapping_notes_in_score,
    control_dtype,
    note_dtype,
    pedal_dtype,
    pitch_bend_dtype,
    Score,
    Track,
)


@dataclass
class PianoRoll:
    """Dataclass for piano rolls."""

    array: np.ndarray
    time_step: float
    pitch_min: int
    num_bin_per_semitone: int

    @property
    def duration(self) -> float:
        return self.array.shape[1] * self.time_step

    @property
    def pitch_max(self) -> int:
        return ceil(self.array.shape[0] / self.num_bin_per_semitone + self.pitch_min)

    def __post_init__(self) -> None:
        assert len(self.array.shape) == 2, "Piano roll must be 2D"
        assert self.array.dtype == np.uint8, "Piano roll must be uint8"
        assert self.array.shape[0] % self.num_bin_per_semitone == 0, (
            "Piano roll shape[0] must be divisible by num_bin_per_semitone"
        )
        assert self.array.shape[1] > 0, "Piano roll shape[1] must be greater than 0"
        assert self.pitch_min >= 0, "pitch_min must be greater than or equal to 0"


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
        if shorten_notes:
            note_end = note_end - 2 * time_step

        col_end_float = note_end / time_step
        alpha_start = 1.0 - (col_start_float - int(col_start_float))
        alpha_end = col_end_float - int(col_end_float)
        row = (note_pitch - pitch_min) * num_bin_per_semitone

        piano_roll[row, int(col_start_float)] += alpha_start * note_velocity
        if alpha_end > 0:
            piano_roll[row, int(col_end_float)] += alpha_end * note_velocity
        for col in range(int(col_start_float) + 1, int(col_end_float)):
            piano_roll[row, col] += note_velocity


def get_piano_roll(
    score: Score,
    time_step: float,
    pitch_min: int,
    pitch_max: int,
    num_bin_per_semitone: int,
    shorten_notes: bool = True,
) -> PianoRoll:
    """Create a piano roll representation of the score.

    use shorten_notes to shorten the notes by 2*time_step to have gaps between notes
    """
    check_no_overlapping_notes_in_score(score)
    num_cols = int(np.ceil(score.duration / time_step))
    num_rows = (pitch_max - pitch_min) * num_bin_per_semitone
    piano_roll = np.zeros((num_rows, num_cols), dtype=float)
    for track in score.tracks:
        _add_notes_to_piano_roll_jit(
            piano_roll=piano_roll,
            pitch=track.notes["pitch"],
            start=track.notes["start"],
            duration=track.notes["duration"],
            velocity=track.notes["velocity"],
            time_step=time_step,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            num_bin_per_semitone=num_bin_per_semitone,
            shorten_notes=shorten_notes,
        )

    return PianoRoll(
        array=piano_roll.astype(np.uint8),
        pitch_min=pitch_min,
        time_step=time_step,
        num_bin_per_semitone=num_bin_per_semitone,
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
    program: int,
    threshold: float = 0.0,
) -> Score:
    """Create a score from a piano roll representation."""
    assert piano_roll.array.dtype == np.uint8, "Piano roll array must be of type uint8"
    piano_roll_semitone = piano_roll.array.reshape(-1, piano_roll.num_bin_per_semitone, piano_roll.array.shape[1]).max(
        axis=1
    )

    start, duration, pitch, velocity = _piano_roll_to_score_jit(
        piano_roll=piano_roll_semitone,
        time_step=piano_roll.time_step,
        pitch_min=piano_roll.pitch_min,
        threshold=threshold,
    )
    # Create the structured array
    notes = np.empty(len(start), dtype=note_dtype)

    # Assign values to the fields
    notes["start"] = start
    notes["duration"] = duration
    notes["pitch"] = pitch
    notes["velocity"] = velocity
    controls = np.array([], dtype=control_dtype)
    pedals = np.array([], dtype=pedal_dtype)
    pitch_bends = np.array([], dtype=pitch_bend_dtype)

    check_no_overlapping_notes(notes)
    track = Track(
        program=program,
        is_drum=False,
        name="Piano",
        notes=notes,
        controls=controls,
        pedals=pedals,
        pitch_bends=pitch_bends,
    )
    tracks = [track]
    return Score(tracks=tracks, duration=piano_roll.duration)
