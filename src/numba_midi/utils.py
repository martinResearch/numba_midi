"""Utility functions for MIDI processing."""

import numpy as np


def is_compound_meter_arrays(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Check if the time signature is a compound meter.
    a signature is a compound meter if all applies:
    * the numerator is divisible by 3
    * the numerator is greater than 3
    * the denominator is 8 or 16.
    """
    return (numerator % 3 == 0) & (numerator > 3) & ((denominator == 8) | (denominator == 16))


def is_compound_meter(numerator: int, denominator: int) -> bool:
    """Check if the time signature is a compound meter.
    A signature is a compound meter if all applies:
    * the numerator is divisible by 3
    * the numerator is greater than 3
    * the denominator is 8 or 16.
    """
    return numerator % 3 == 0 and numerator > 3 and (denominator in (8, 16))


def get_beats_per_bar(numerator: int, denominator: int) -> int:
    if is_compound_meter(numerator, denominator):
        # Assume it's compound meter
        return numerator // 3
    else:
        # Simple meter
        return numerator


def get_quarter_notes_per_beat(numerator: int, denominator: int) -> float:
    """
    Compute how many quarter notes are in one beat,
    based on the time signature.

    Args:
        numerator (int): top number of time signature (e.g., 12 in 12/8)
        denominator (int): bottom number of time signature (e.g., 8 in 12/8)

    Returns:
        float: number of quarter notes per beat
    """
    # 1 beat = note value defined by denominator
    note_value_in_quarter_notes = 4 / denominator

    # Detect compound meter (e.g., 6/8, 9/8, 12/8)
    if is_compound_meter(numerator, denominator):
        # Each beat = 3 eighth notes = 1.5 quarter notes
        return 1.5
    else:
        # Simple meter: 1 beat = denominator note value
        return note_value_in_quarter_notes


def get_bpm_from_quarter_notes_per_minute(quarter_notes_per_minute: float, numerator: int, denominator: int) -> float:
    beats_per_minute = quarter_notes_per_minute / get_quarter_notes_per_beat(numerator, denominator)
    return beats_per_minute


def get_bar_duration(quarter_notes_per_minute: float, numerator: int, denominator: int) -> float:
    beats_per_bar = get_beats_per_bar(numerator, denominator)
    return (
        60.0 / get_bpm_from_quarter_notes_per_minute(quarter_notes_per_minute, numerator, denominator) * beats_per_bar
    )


def get_tick_per_beat(ticks_per_quarter: int, numerator: int, denominator: int) -> float:
    ticks_per_beat = ticks_per_quarter * get_quarter_notes_per_beat(numerator, denominator)
    return ticks_per_beat


def get_quarter_notes_per_beat_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    # 1 beat = note value defined by denominator
    note_value_in_quarter_notes = 4 / denominator
    # Detect compound meter (e.g., 6/8, 9/8, 12/8)
    is_compound_meter = is_compound_meter_arrays(numerator, denominator)
    # Each beat = 3 eighth notes = 1.5 quarter notes
    return np.where(is_compound_meter, 1.5, note_value_in_quarter_notes)


def get_tick_per_beat_array(ticks_per_quarter: int, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return ticks_per_quarter * get_quarter_notes_per_beat_array(numerator, denominator)
