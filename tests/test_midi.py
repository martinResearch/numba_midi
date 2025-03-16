"""Test the numba_midi library."""

import glob
from pathlib import Path
import shutil

import numpy as np
import symusic
import tqdm

from numba_midi import load_score
from numba_midi.midi import (
    assert_midi_equal,
    load_midi_bytes,
    load_midi_score,
    save_midi_data,
    sort_midi_events,
)
from numba_midi.score import assert_scores_equal, midi_to_score, score_to_midi


def test_numba_midi() -> None:
    midi_file = Path(__file__).parent / "data" / "b81b22b84dfd54e2aead3f0207889d38.mid"
    assert midi_file.exists()
    score_numba = load_score(midi_file)
    assert score_numba.duration == 289.08755

    assert score_numba.tracks[0].name == "HitBit  "
    assert len(score_numba.tracks[0].notes) == 405
    assert score_numba.tracks[0].notes[0]["pitch"] == 76
    assert score_numba.tracks[0].notes[0]["start"] == 20.261805
    assert score_numba.tracks[0].notes[0]["duration"] == 0.7643566


def test_save_midi_data_load_midi_bytes_roundtrip() -> None:
    midi_files = glob.glob(str(Path(__file__).parent / "data" / "*.mid"))
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
    midi_files = glob.glob(str(Path(__file__).parent / "data" / "*.mid"))
    for midi_file in midi_files:
        print(f"Testing sort_midi_events with {midi_file}")
        # load row midi score
        midi_raw = load_midi_score(midi_file)
        sorted_events = sort_midi_events(midi_raw.tracks[0].events)
        sorted_events2 = sort_midi_events(sorted_events)
        assert np.all(sorted_events == sorted_events2)


def get_lakh_dataset_failure_cases() -> None:
    midi_files = glob.glob(
        str(Path(r"C:\repos\audio_to_midi\src\audio_to_midi\datasets\lakh_midi\lmd_matched") / "**" / "*.mid"),
        recursive=True,
    )
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

            # compare with symusic
            symusic.Score.from_file(midi_file)
            # symusic_tracks_with_notes = [track for track in symusic_score_ticks.tracks if len(track.notes) > 0]
            # for i, track in enumerate(score.tracks):
            #     symusic_track_ticks = symusic_tracks_with_notes[i]
            #     assert len(track.notes) == len(symusic_track_ticks.notes)
            #     symusic_notes_numpy = symusic_track_ticks.notes.numpy()
            #     assert np.all(track.notes["pitch"] == symusic_notes_numpy["pitch"])
            #     assert np.all(track.notes["start_tick"] == symusic_notes_numpy["time"])
            #     assert np.all(track.notes["duration_tick"] == symusic_notes_numpy["duration"])
            # symusic_track_sec = symusic_score_sec.tracks[i]
            # symusic_notes_sec_numpy = symusic_track_sec.notes.numpy()
            # assert np.allclose(track.notes["start"], symusic_notes_sec_numpy["time"], 1e-4)
            # assert np.allclose(track.notes["duration"], symusic_notes_sec_numpy["duration"], atol=1e-5)

            midi_raw2 = score_to_midi(score)
            score2 = midi_to_score(midi_raw2)
            # check if the two midi scores are equal
            assert_scores_equal(score, score2)
        except Exception as e:
            print(f"Failed to process {midi_file}: {e}")
            # copy the faild ons=es to the Path(__file__).parent / "data"  folder
            # do not do a rename
            shutil.copy(midi_file, Path(__file__).parent / "data" / Path(midi_file).name)


# def load_score_symusic(midi_file: str):
#     symusic_score_ticks = symusic.Score.from_file(midi_file)
#     tracks=[]
#     for i, track in enumerate(symusic_score_ticks.tracks):
#         symusic_track_ticks = symusic_score_ticks.tracks[i]
#         symusic_notes_numpy = symusic_track_ticks.notes.numpy()
#         tracks.append(Track(
#             name=track.name,
#             notes={
#                 "pitch": symusic_notes_numpy["pitch"],
#                 "start_tick": symusic_notes_numpy["time"],
#                 "duration_tick": symusic_notes_numpy["duration"],
#             },
#         ))


def test_score_to_midi_midi_to_score_round_trip() -> None:
    midi_files = glob.glob(str(Path(__file__).parent / "data" / "*.mid"))

    midi_files = sorted(midi_files)
    for midi_file in tqdm.tqdm(midi_files):
        # print(f"Testing score_to_midi_midi_to_score_round_trip with {midi_file}")
        # midi_file ="C:\\repos\\audio_to_midi\\src\\audio_to_midi\\datasets\\lakh_midi\\lmd_matched\\A\\A\\A\\TRAAAGR128F425B14B\\5dd29e99ed7bd3cc0c5177a6e9de22ea.mid"
        # symusic_score_ticks = symusic.Score.from_file(midi_file)
        # symusic_score_sec = symusic.Score.from_file(midi_file, ttype="second")
        # tinysoundfont_score = tinysoundfont.midi.load(midi_file)
        # load row midi score
        # [
        #     event
        #     for event in tinysoundfont_score
        #     if event.channel == 0 and isinstance(event.action, tinysoundfont.midi.NoteOn) and event.action.key == 65
        # ]

        midi_raw = load_midi_score(midi_file)

        score = midi_to_score(midi_raw)
        midi_raw2 = score_to_midi(score)
        score2 = midi_to_score(midi_raw2)

        # check if the two scores are equal
        assert_scores_equal(score, score2)

        # compare with symusic
        symusic_score_ticks = symusic.Score.from_file(midi_file)
        symusic_tracks_with_notes = [track for track in symusic_score_ticks.tracks if len(track.notes) > 0]
        assert len(score.tracks) == len(symusic_tracks_with_notes)
        for i, track in enumerate(score.tracks):
            symusic_track_ticks = symusic_tracks_with_notes[i]
            assert len(track.notes) == len(symusic_track_ticks.notes)
            symusic_notes_numpy = symusic_track_ticks.notes.numpy()
            assert np.all(track.notes["pitch"] == symusic_notes_numpy["pitch"])
            assert np.all(track.notes["start_tick"] == symusic_notes_numpy["time"])
            assert np.all(track.notes["duration_tick"] == symusic_notes_numpy["duration"])
            # symusic_track_sec = symusic_score_sec.tracks[i]
            # symusic_notes_sec_numpy = symusic_track_sec.notes.numpy()
            # assert np.allclose(track.notes["start"], symusic_notes_sec_numpy["time"], 1e-4)
            # assert np.allclose(track.notes["duration"], symusic_notes_sec_numpy["duration"], atol=1e-5)


if __name__ == "__main__":
    get_lakh_dataset_failure_cases()
    # test_score_to_midi_midi_to_score_round_trip()
    # test_sort_midi_events()

    # test_numba_midi()
    # test_save_midi_data_load_midi_bytes_roundtrip()
    # print("All tests passed!")
