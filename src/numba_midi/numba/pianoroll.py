"""Tools to process piano roll reprentationss of MIDI scores."""

from numba.core.decorators import njit
import numpy as np


@njit(cache=True, boundscheck=False, nogil=True, fastmath=True)
def add_notes_to_piano_roll(
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


@njit(cache=True, boundscheck=False, nogil=True, fastmath=True)
def piano_roll_to_score(
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
