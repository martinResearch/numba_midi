"""Test the numba_midi library."""

from pathlib import Path

from numba_midi import load_score
from numba_midi.midi import assert_midi_equal, load_midi_bytes, load_midi_score, save_midi_data
from numba_midi.score import assert_scores_equal, midi_to_score, score_to_midi


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


def test_score_to_midi_midi_to_score_round_trip() -> None:
    midi_file = Path(__file__).parent / "data" / "b81b22b84dfd54e2aead3f0207889d38.mid"
    assert midi_file.exists()

    # load row midi score
    score = load_score(midi_file)

    midi_raw = score_to_midi(score)
    score2 = midi_to_score(midi_raw)

    # check if the two midi scores are equal
    assert_scores_equal(score, score2)


if __name__ == "__main__":
    test_numba_midi()
    test_save_midi_data_load_midi_bytes_roundtrip()
    test_score_to_midi_midi_to_score_round_trip()
