"""Utility functions for MIDI processing."""

import numpy as np


def get_beats_per_bar(numerator: int, denominator: int) -> int:
    if numerator % 3 == 0 and denominator == 8:
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
    if numerator % 3 == 0 and denominator == 8:
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
    is_compound_meter = np.logical_and(numerator % 3 == 0, denominator == 8)
    # Each beat = 3 eighth notes = 1.5 quarter notes
    return np.where(is_compound_meter, 1.5, note_value_in_quarter_notes)


def get_tick_per_beat_array(ticks_per_quarter: int, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return ticks_per_quarter * get_quarter_notes_per_beat_array(numerator, denominator)
