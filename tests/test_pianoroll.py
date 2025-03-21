"""Test the conversion between pianoroll and Score objects."""

import glob
from pathlib import Path

from numba_midi.pianoroll import piano_roll_to_score, score_to_piano_roll
from numba_midi.score import assert_scores_equal, load_score, remove_control_changes, remove_pitch_bends


def test_pianoroll_conversion() -> None:
    """Test pianoroll conversion functions."""
    midi_files = glob.glob(str(Path(__file__).parent / "data" / "numba_midi" / "*.mid"))

    for midi_file in midi_files:
        score1 = load_score(midi_file, notes_mode=1, minimize_tempo=False)

        # Convert to pianoroll object
        time_step = 60 / (score1.tempo[0]["bpm"] * score1.clocks_per_click)
        pianoroll = score_to_piano_roll(score1, time_step=time_step, pitch_min=0, pitch_max=128, num_bin_per_semitone=1)

        # Convert back to Score object
        score2 = piano_roll_to_score(pianoroll)

        # remove control changes and  pitch bends
        score1 = remove_pitch_bends(remove_control_changes(score1))
        assert_scores_equal(
            score1,
            score2,
            sort_tracks_with_programs_and_num_notes=True,
            tick_tol=100,
            time_tol=3 * time_step,
            value_tol=100.0,
        )


if __name__ == "__main__":
    test_pianoroll_conversion()
