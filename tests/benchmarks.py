import glob
from pathlib import Path

import symusic
import pretty_midi

from numba_midi.interop.symusic import from_symusic, to_symusic
from numba_midi.score import assert_scores_equal, load_score
import time

def benchmark():

    # Create a PrettyMIDI object
    midi_files = glob.glob(str(Path(__file__).parent / "data" / "symusic" / "*.mid"))
    num_iterations=100
    for midi_file in midi_files:
        # benchmark with numba_midi
        durarions = []
        for k in range(num_iterations):
            start_time = time.perf_counter()
            # load the score using numba_midi
            score1 = load_score(midi_file, notes_mode=3, minimize_tempo=False)
            end_time = time.perf_counter()
            durarions.append(end_time - start_time)
        min_duration = min(durarions)
        avg_duration = sum(durarions) / num_iterations
        print(f"Min duration for {midi_file}: {1000*min_duration:.5f} milliseconds")
        # Load the MIDI file
        symusic_score = symusic.Score.from_file(midi_file)

        # benchmark with symusic
        durarions = []
        for k in range(num_iterations):
            start_time = time.perf_counter()
            # Convert to Score object
            score2 = from_symusic(symusic_score)
            end_time = time.perf_counter()
            durarions.append(end_time - start_time)
        min_duration = min(durarions)
        avg_duration = sum(durarions) / num_iterations
        print(f"Min duration for {midi_file}: {1000*min_duration:.5f} milliseconds")

        # benchmark with pretty_midi
        durarions = []
        for k in range(num_iterations):
            start_time = time.perf_counter()
            # Convert to Score object
            midi = pretty_midi.PrettyMIDI(midi_file)
            end_time = time.perf_counter()
            durarions.append(end_time - start_time)
        min_duration = min(durarions)
        avg_duration = sum(durarions) / num_iterations
        print(f"Min duration for {midi_file}: {1000*min_duration:.5f} milliseconds")
if __name__ == "__main__":
    benchmark()