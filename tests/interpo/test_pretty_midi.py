"""Test PrettyMIDI conversion functions."""

import glob
from pathlib import Path

import pretty_midi

from numba_midi.interop.pretty_midi import from_pretty_midi, to_pretty_midi
from numba_midi.score import assert_scores_equal, load_score


def test_pretty_midi_conversion() -> None:
    # Create a PrettyMIDI object
    midi_files = glob.glob(str(Path(__file__).parent.parent / "data" / "pretty_midi"/"*.mid"))
    for midi_file in midi_files:
        print(f"Testing PrettyMIDI conversion with {midi_file}")
        # load the score uing numba_midi
        score1 = load_score(midi_file, notes_mode=5, minimize_tempo=True)

        # Load the MIDI file
        midi = pretty_midi.PrettyMIDI(midi_file)

        # Convert to Score object
        score2 = from_pretty_midi(midi)

        (
            assert_scores_equal(score1, score2, sort_tracks_with_programs_and_num_notes=True),
            "The original and converted Score objects are not equal",
        )

        # Convert back to PrettyMIDI object
        midi_converted = to_pretty_midi(score2)

        # Convert back to Score object
        score3 = from_pretty_midi(midi_converted)
        (
            assert_scores_equal(score1, score3, sort_tracks_with_programs_and_num_notes=True),
            "The original and converted Score objects are not equal",
        )


if __name__ == "__main__":
    test_pretty_midi_conversion()
    print("PrettyMIDI conversion test passed.")
