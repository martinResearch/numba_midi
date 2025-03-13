"""Test the numba_midi library."""

from pathlib import Path

import numpy as np

from numba_midi import load_score
from numba_midi.midi import (
    assert_midi_equal,
    load_midi_bytes,
    load_midi_score,
    save_midi_data,
    sort_midi_events,
)
from numba_midi.score import assert_scores_equal, check_no_overlapping_notes, midi_to_score, score_to_midi


def test_numba_midi() -> None:
    midi_file = Path(__file__).parent / "data" / "b81b22b84dfd54e2aead3f0207889d38.mid"
    assert midi_file.exists()
    score_numba = load_score(midi_file)
    assert score_numba.duration == 289.0495300292969

    assert score_numba.tracks[0].name == "HitBit  "
    assert len(score_numba.tracks[0].notes) == 405
    assert score_numba.tracks[0].notes[0]["pitch"] == 76
    assert score_numba.tracks[0].notes[0]["start"] == 20.223797
    assert score_numba.tracks[0].notes[0]["duration"] == 0.7643585


def test_save_midi_data_load_midi_bytes_roundtrip() -> None:
    midi_file = Path(__file__).parent / "data" / "b81b22b84dfd54e2aead3f0207889d38.mid"
    assert midi_file.exists()

    # load row midi score
    midi_raw = load_midi_score(midi_file)

    # save to bytes and load it back
    data2 = save_midi_data(midi_raw)
    midi_raw2 = load_midi_bytes(data2)

    # check if the two midi scores are equal
    assert_midi_equal(midi_raw, midi_raw2)


def test_sort_midi_events() -> None:
    midi_file = Path(__file__).parent / "data" / "b81b22b84dfd54e2aead3f0207889d38.mid"
    assert midi_file.exists()
    # load row midi score
    midi_raw = load_midi_score(midi_file)
    sorted_events = sort_midi_events(midi_raw.tracks[0].events)
    sorted_events2 = sort_midi_events(sorted_events)
    assert np.all(sorted_events == sorted_events2)


def test_score_to_midi_midi_to_score_round_trip() -> None:
    midi_file = Path(__file__).parent / "data" / "b81b22b84dfd54e2aead3f0207889d38.mid"
    assert midi_file.exists()

    # load row midi score
    midi_raw = load_midi_score(midi_file)
    score = midi_to_score(midi_raw)
    check_no_overlapping_notes(score.tracks[0].notes, use_ticks=True)

    midi_raw2 = score_to_midi(score)
    score2 = midi_to_score(midi_raw2)

    # check if the two scores are equal
    assert_scores_equal(score, score2)


if __name__ == "__main__":
    test_score_to_midi_midi_to_score_round_trip()
    test_sort_midi_events()

    test_numba_midi()
    test_save_midi_data_load_midi_bytes_roundtrip()
    print("All tests passed!")
