"""Tests for the score module in numba_midi."""

import glob
from pathlib import Path

import numpy as np

from numba_midi.score import load_score


def test_get_beat_positions() -> None:
    """Test the get_beat_positions method of the Score class."""
    midi_files = glob.glob(str(Path(__file__).parent / "data" / "numba_midi" / "*.mid"))
    for midi_file in midi_files:
        print(f"Testing get_beat_positions with {midi_file}")
        # load row midi score
        score = load_score(midi_file)
        beat_positions = score.get_beat_positions()
        assert len(beat_positions) > 0
        assert np.all(np.diff(beat_positions) > 0)
        bar = score.get_bar_positions()
        assert len(bar) > 0
        assert np.all(np.diff(bar) > 0)


if __name__ == "__main__":
    test_get_beat_positions()
    print("All tests passed.")
