"""Test the numba_midi library."""

import glob
from pathlib import Path
import shutil

import numpy as np
import symusic
import tqdm

from numba_midi.midi import (
    assert_midi_equal,
    load_midi_bytes,
    load_midi_score,
    save_midi_data,
    sort_midi_events,
)
from numba_midi.score import assert_scores_equal, midi_to_score, score_to_midi


def test_save_midi_data_load_midi_bytes_roundtrip() -> None:
    midi_files = glob.glob(str(Path(__file__).parent / "data" / "numba_midi" / "*.mid"))
    for midi_file in midi_files:
        print(f"Testing save_midi_data_load_midi_bytes_roundtrip with {midi_file}")
        # load row midi score
        midi_raw = load_midi_score(midi_file)

        # save to bytes and load it back
        data2 = save_midi_data(midi_raw)
        midi_raw2 = load_midi_bytes(data2)

        # check if the two midi scores are equal
        assert_midi_equal(midi_raw, midi_raw2)


def test_sort_midi_events() -> None:
    midi_files = glob.glob(str(Path(__file__).parent / "data" / "numba_midi" / "*.mid"))
    for midi_file in midi_files:
        print(f"Testing sort_midi_events with {midi_file}")
        # load row midi score
        midi_raw = load_midi_score(midi_file)
        sorted_events = sort_midi_events(midi_raw.tracks[0].events)
        sorted_events2 = sort_midi_events(sorted_events)
        assert np.all(sorted_events._data == sorted_events2._data)


def collect_lakh_dataset_failure_cases(compare_to_symusic: bool = False) -> None:
    midi_files = glob.glob(
        str(Path(r"C:\repos\audio_to_midi\src\audio_to_midi\datasets\lakh_midi\lmd_matched") / "**" / "*.mid"),
        recursive=True,
    )
    midi_files = midi_files[:10]
    for midi_file in tqdm.tqdm(midi_files):
        try:
            symusic.Score.from_file(midi_file)
        except Exception:
            continue
        try:
            # load row midi score
            midi_raw = load_midi_score(midi_file)
            # save to bytes and load it back
            score = midi_to_score(midi_raw)

            midi_raw2 = score_to_midi(score)
            score2 = midi_to_score(midi_raw2)
            # check if the two midi scores are equal
            assert_scores_equal(score, score2)
        except Exception as e:
            print(f"Failed to process {midi_file}: {e}")
            # copy the faild ons=es to the Path(__file__).parent / "data"  folder
            # do not do a rename
            shutil.copy(midi_file, Path(__file__).parent / "data" / "numba_midi" / Path(midi_file).name)


def test_score_to_midi_midi_to_score_round_trip(compare_to_symusic: bool = False) -> None:
    midi_files = glob.glob(str(Path(__file__).parent / "data" / "numba_midi" / "*.mid"))

    midi_files = sorted(midi_files)
    for midi_file in tqdm.tqdm(midi_files):
        midi_raw = load_midi_score(midi_file)
        score = midi_to_score(midi_raw)
        midi_raw2 = score_to_midi(score)
        score2 = midi_to_score(midi_raw2)

        # check if the two scores are equal
        assert_scores_equal(score, score2)


if __name__ == "__main__":
    # collect_lakh_dataset_failure_cases()
    test_score_to_midi_midi_to_score_round_trip()
    test_sort_midi_events()
    test_save_midi_data_load_midi_bytes_roundtrip()
    print("All tests passed!")
