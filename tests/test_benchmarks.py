"""Benchmarking script for loading MIDI files using numba_midi, symusic, and pretty_midi."""

import glob
from pathlib import Path
import time

import pretty_midi
import symusic

from numba_midi.score import load_score


def test_benchmark() -> None:
    # Create a PrettyMIDI object
    midi_files = glob.glob(str(Path(__file__).parent / "data" / "symusic" / "*.mid"))
    num_iterations = 100
    for midi_file in midi_files:
        # benchmark with numba_midi
        durations = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            # load the score using numba_midi
            load_score(midi_file, notes_mode="note_off_stops_all", minimize_tempo=False)
            end_time = time.perf_counter()
            durations.append(end_time - start_time)
        min_duration = min(durations)

        print(f"Min duration for {midi_file}: {1000 * min_duration:.5f} milliseconds")

        # benchmark with symusic
        durations = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            symusic.Score.from_file(midi_file)
            end_time = time.perf_counter()
            durations.append(end_time - start_time)
        min_duration = min(durations)

        print(f"Min duration for {midi_file}: {1000 * min_duration:.5f} milliseconds")

        # benchmark with pretty_midi
        durations = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            # Convert to Score object
            pretty_midi.PrettyMIDI(midi_file)
            end_time = time.perf_counter()
            durations.append(end_time - start_time)
        min_duration = min(durations)

        print(f"Min duration for {midi_file}: {1000 * min_duration:.5f} milliseconds")


if __name__ == "__main__":
    test_benchmark()
