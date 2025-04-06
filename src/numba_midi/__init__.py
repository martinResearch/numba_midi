"""A numba-accelerated python midi score processing library."""

from numba_midi.score import load_score, load_score_bytes

__all__ = ["load_score", "load_score_bytes"]
