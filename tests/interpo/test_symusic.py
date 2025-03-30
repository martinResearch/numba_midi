"""Test symusic conversion functions."""

import glob
from pathlib import Path

import symusic

from numba_midi.interop.symusic import from_symusic, to_symusic
from numba_midi.score import assert_scores_equal, load_score


def test_symusic_conversion() -> None:
    # Create a PrettyMIDI object
    midi_files = glob.glob(str(Path(__file__).parent.parent / "data" / "symusic" / "*.mid"))
    for midi_file in midi_files:
        print(f"Testing PrettyMIDI conversion with {midi_file}")
        # load the score uing numba_midi
        score1 = load_score(midi_file, notes_mode=3, minimize_tempo=False)

        # Load the MIDI file
        symusic_score = symusic.Score.from_file(midi_file)

        # Convert to Score object
        score2 = from_symusic(symusic_score)

        (
            assert_scores_equal(score1, score2, compare_channels=False),
            "The original and converted Score objects are not equal",
        )

        # Convert back to symusic object
        symusic_score_converted = to_symusic(score2)

        # Convert back to Score object
        score3 = from_symusic(symusic_score_converted)
        (
            assert_scores_equal(score1, score3, compare_channels=False),
            "The original and converted Score objects are not equal",
        )


if __name__ == "__main__":
    test_symusic_conversion()
    print("PrettyMIDI conversion test passed.")
