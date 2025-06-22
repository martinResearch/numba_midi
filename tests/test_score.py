"""Tests for the score module in numba_midi."""

import glob
from pathlib import Path

import numpy as np

from numba_midi.score import (
    ControlArray,
    load_score,
    PedalArray,
    PitchBendArray,
    Score,
    SignatureArray,
    TempoArray,
    Track,
)
from numba_midi.utils import get_bar_duration, get_bpm_from_quarter_notes_per_minute


def test_get_beat_positions() -> None:
    """Test the get_beat_positions method of the Score class."""
    midi_files = glob.glob(str(Path(__file__).parent / "data" / "numba_midi" / "*.mid"))
    for midi_file in midi_files:
        print(f"Testing get_beat_positions with {midi_file}")
        # load row midi score
        score = load_score(midi_file)
        subdivisions, beats, bars = score.get_subdivision_beat_and_bar_times()
        assert len(subdivisions) > 0
        assert np.all(np.diff(subdivisions) > 0)
        assert len(beats) > 0
        assert np.all(np.diff(beats) > 0)
        assert len(bars) > 0
        assert np.all(np.diff(bars) > 0)


def test_signature() -> None:
    # in midi the bmp
    signatures = [(12, 8), (4, 4), (2, 2), (2, 4), (3, 4), (3, 8), (6, 8), (9, 8)]

    quarter_notes_per_minute = 60.0
    for signature in signatures:
        numerator, denominator = signature

        print(f"Testing signature with numerator={numerator}, denominator={denominator}")
        time_signature = SignatureArray(
            numerator=[numerator],
            denominator=[denominator],
            tick=[0],
            time=[0.0],
            clocks_per_click=[24],
            notated_32nd_notes_per_beat=[8],
        )
        tempo = TempoArray(
            tick=[0],
            time=[0.0],
            quarter_notes_per_minute=[quarter_notes_per_minute],
        )
        score = Score(ticks_per_quarter=480, time_signature=time_signature, tempo=tempo, last_tick=480 * 10, tracks=[])
        notes = score.create_notes(
            start=np.array([0]),
            duration=np.array([1]),
            pitch=np.array([60]),
            velocity=np.array([100]),
        )

        track = Track(
            program=0,
            is_drum=False,
            name="Track 1",
            notes=notes,
            controls=ControlArray.zeros(0),
            pitch_bends=PitchBendArray.zeros(0),
            pedals=PedalArray.zeros(0),
        )
        score.add_track(track)

        print(score)
        subdivisions, beats, bars = score.get_subdivision_beat_and_bar_times()
        print("Beat positions:", beats)
        print("Bar positions:", bars)
        assert len(beats) > 0
        assert len(bars) > 0

        # calculating the musical BPM i.e number of beat per minute and
        # taking the signature denominator into account
        # While people often confusingly refer to the number of quarter
        # notes per minute as the BPM

        expected_bpm = get_bpm_from_quarter_notes_per_minute(quarter_notes_per_minute, numerator, denominator)
        bpm = 60.0 / (beats[1] - beats[0])
        assert np.isclose(bpm, expected_bpm), f"Expected {expected_bpm}, got {bpm}"

        expected_bar_duration = get_bar_duration(quarter_notes_per_minute, numerator, denominator)
        assert np.isclose(bars[1] - bars[0], expected_bar_duration), (
            f"Expected {expected_bar_duration}, got {bars[1] - bars[0]}"
        )


if __name__ == "__main__":
    test_signature()
    test_get_beat_positions()
    print("All tests passed.")
