# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

"""
Cython implementations of pianoroll operations.
Replaces numba-jitted functions in pianoroll.py for better performance and precompiled distribution.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport ceil

# Initialize numpy
cnp.import_array()

def add_notes_to_piano_roll(
    cnp.ndarray[cnp.double_t, ndim=2] piano_roll,
    cnp.ndarray[cnp.int32_t, ndim=1] pitch,
    cnp.ndarray[cnp.float64_t, ndim=1] start,
    cnp.ndarray[cnp.float64_t, ndim=1] duration,
    cnp.ndarray[cnp.uint8_t, ndim=1] velocity,
    double time_step,
    int pitch_min,
    int pitch_max,
    int num_bin_per_semitone,
    bint shorten_notes,
    bint antialiasing = False,
):
    """
    Add notes to a piano roll representation.
    
    This function fills the piano roll array with note information.
    Each note is represented by its pitch, start time, duration, and velocity.
    """
    cdef Py_ssize_t note_id, row, col
    cdef Py_ssize_t num_notes = len(pitch)
    cdef cnp.int32_t note_pitch
    cdef cnp.uint8_t note_velocity
    cdef cnp.float32_t note_start, note_duration, note_end
    cdef double col_start_float, col_end_float
    cdef int col_start_int, col_end_int
    cdef double alpha_start, alpha_end

    for note_id in range(num_notes):
        note_pitch = pitch[note_id]
        note_start = start[note_id]
        note_duration = duration[note_id]
        note_velocity = velocity[note_id]

        if note_pitch < pitch_min or note_pitch >= pitch_max:
            continue  # Skip notes outside the pitch range

        col_start_float = note_start / time_step
        note_end = note_start + note_duration
        row = (note_pitch - pitch_min) * num_bin_per_semitone
        col_end_float = note_end / time_step

        if antialiasing:
            if shorten_notes:
                note_end = note_end - 2 * time_step
                col_end_float = note_end / time_step

            alpha_start = 1.0 - (col_start_float - <int>col_start_float)
            alpha_end = col_end_float - <int>col_end_float

            # Add antialiased start
            piano_roll[row, <int>col_start_float] += alpha_start * note_velocity
            
            # Add antialiased end
            if alpha_end > 0:
                piano_roll[row, <int>col_end_float] += alpha_end * note_velocity

            # Fill the middle columns
            for col in range(<int>col_start_float + 1, <int>col_end_float):
                piano_roll[row, col] += note_velocity
        else:
            col_start_int = <int>col_start_float
            col_end_int = <int>col_end_float

            if shorten_notes:
                col_end_int -= 1

            # Fill the columns with the note
            for col in range(col_start_int, col_end_int):
                if col >= 0 and col < piano_roll.shape[1]:  # Bounds check
                    piano_roll[row, col] += note_velocity


def piano_roll_to_score(
    cnp.ndarray[cnp.uint8_t, ndim=2] piano_roll,
    double time_step,
    int pitch_min = 0,
    double threshold = 0.0
):
    """
    Convert a piano roll representation back to score format.
    
    Returns
    -------
    tuple
        (start, duration, pitch, velocity) arrays
    """
    cdef list start_list = []
    cdef list duration_list = []
    cdef list pitch_list = []
    cdef list velocity_list = []
    
    cdef Py_ssize_t row, col
    cdef int note_pitch
    cdef bint note_on
    cdef int sum_velocity, col_start, duration_col
    cdef double note_start, note_duration
    cdef cnp.uint8_t note_velocity
    
    cdef Py_ssize_t num_rows = piano_roll.shape[0]
    cdef Py_ssize_t num_cols = piano_roll.shape[1]

    for row in range(num_rows):
        note_pitch = pitch_min + row
        note_on = False
        sum_velocity = 0
        col_start = 0

        for col in range(num_cols):
            if piano_roll[row, col] > threshold:
                if not note_on:
                    col_start = col
                    note_on = True
                    sum_velocity = <int>piano_roll[row, col]
                else:
                    sum_velocity += <int>piano_roll[row, col]
            elif note_on:
                # Note ended
                note_start = col_start * time_step
                duration_col = col - col_start
                note_duration = duration_col * time_step
                note_velocity = <cnp.uint8_t>(sum_velocity / duration_col)
                note_on = False

                start_list.append(note_start)
                duration_list.append(note_duration)
                pitch_list.append(note_pitch)
                velocity_list.append(note_velocity)

        # Handle note still on at the end
        if note_on:
            note_start = col_start * time_step
            duration_col = num_cols - col_start
            note_duration = duration_col * time_step
            note_velocity = <cnp.uint8_t>(sum_velocity / duration_col)

            start_list.append(note_start)
            duration_list.append(note_duration)
            pitch_list.append(note_pitch)
            velocity_list.append(note_velocity)

    # Convert lists to numpy arrays
    cdef cnp.ndarray[cnp.float32_t, ndim=1] start_array = np.array(start_list, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] duration_array = np.array(duration_list, dtype=np.float32)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] pitch_array = np.array(pitch_list, dtype=np.uint8)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] velocity_array = np.array(velocity_list, dtype=np.uint8)

    return start_array, duration_array, pitch_array, velocity_array
