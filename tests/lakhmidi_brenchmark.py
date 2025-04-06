"""Benchmark  MIDI loading times for different libraries using the Lakh dataset."""

import glob
from pathlib import Path
import tarfile
import time
from typing import Any, Callable, Literal, overload

import pandas as pd
import pretty_midi
import requests
import symusic
import tqdm

import numba_midi


def decompress_tar_gz_with_progress(file_path: str, output_dir: str) -> None:
    with tarfile.open(file_path, "r:gz") as tar:
        members = tar.getmembers()
        total_files = len(members)
        with tqdm.tqdm(total=total_files, desc="Decompressing", unit="file") as pbar:
            for member in members:
                tar.extract(member, path=output_dir)
                pbar.update(1)


def download_dataset() -> None:
    folder = Path(__file__).parent / "data" / "lakh"
    folder.mkdir(parents=True, exist_ok=True)
    url = "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz"
    print(f"Downloading Lakh MIDI Dataset from {url}...")
    file = folder / "lmd_matched.tar.gz"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def decompress_dataset() -> None:
    folder = Path(__file__).parent / "data" / "lakh"
    file = folder / "lmd_matched.tar.gz"
    # decompress with progress bar
    decompress_tar_gz_with_progress(str(file), str(folder))


@overload
def benchmark_method(
    name: str,
    func: Callable[[bytes], Any],
    num_max_files: int,
    num_iter: int,
    input_bytes: Literal[True],
) -> None: ...


@overload
def benchmark_method(
    name: str,
    func: Callable[[str], Any],
    num_max_files: int,
    num_iter: int,
    input_bytes: Literal[False],
) -> None: ...


def benchmark_method(
    name: str,
    func: Callable[[Any], Any],
    num_max_files: int,
    num_iter: int,
    input_bytes: bool,
) -> None:
    # get the list of midi files
    folder = Path(__file__).parent / "data" / "lakh" / "benchmarks"
    folder.mkdir(parents=True, exist_ok=True)
    print("Getting list of MIDI files...")
    midi_files = glob.glob(
        str(Path(r"C:\repos\audio_to_midi\src\audio_to_midi\datasets\lakh_midi\lmd_matched") / "**" / "*.mid"),
        recursive=True,
    )
    midi_files = sorted(midi_files)
    # computem md5 hash of the files names
    # md5_hashes = [Path(midi_file).name for midi_file in midi_files]

    print("Number of MIDI files:", len(midi_files))
    failures = []

    if num_max_files > 0:
        midi_files = midi_files[:num_max_files]
    print(f"Benchmarking {name}...")

    rows = []
    for midi_file in tqdm.tqdm(midi_files):
        filesize = Path(midi_file).stat().st_size
        filesize_mb = filesize / (1024 * 1024)  # Convert to MB
        try:
            # load row midi score
            if input_bytes:
                with open(midi_file, "rb") as file:
                    data = file.read()

                durations = []
                for _ in range(num_iter):
                    start_time = time.perf_counter()
                    func(data)
                    durations.append(time.perf_counter() - start_time)
                min_duration = min(durations)
            else:
                durations = []
                for _ in range(num_iter):
                    start_time = time.perf_counter()
                    func(midi_file)
                    durations.append(time.perf_counter() - start_time)
                min_duration = min(durations)
            rows.append(
                {
                    "file": midi_file,
                    "duration": min_duration,
                    "filesize": filesize,
                    "megabyte_per_second": filesize_mb / min_duration,
                }
            )
        except Exception:
            failures.append(midi_file)

    df = pd.DataFrame(rows)
    df.to_csv(f"{folder}/benchmark_{name}.csv", index=False)
    print("Number of failures:", len(failures))
    df_failures = pd.DataFrame(failures, columns=["file"])
    df_failures.to_csv(f"{folder}/failures_{name}.csv", index=False)

    print("Median bitrate:", df["megabyte_per_second"].median())
    print("Mean bitrate:", df["megabyte_per_second"].mean())
    print("Failure percentage:", len(failures) / len(midi_files) * 100)


def benchmark() -> None:
    num_max_files = 1000
    num_iter = 10
    benchmark_method(
        "pretty_midi", pretty_midi.PrettyMIDI, num_max_files=num_max_files, input_bytes=False, num_iter=num_iter
    )
    benchmark_method(
        "numba_midi_files", numba_midi.load_score, num_max_files=num_max_files, input_bytes=False, num_iter=num_iter
    )
    benchmark_method(
        "numba_midi_bytes",
        numba_midi.load_score_bytes,
        num_max_files=num_max_files,
        input_bytes=True,
        num_iter=num_iter,
    )

    benchmark_method(
        "symusic_files", symusic.Score.from_file, num_max_files=num_max_files, input_bytes=False, num_iter=num_iter
    )
    benchmark_method(
        "symusic_bytes", symusic.Score.from_midi, num_max_files=num_max_files, input_bytes=True, num_iter=num_iter
    )


if __name__ == "__main__":
    # download_dataset()
    # decompress_dataset()
    benchmark()
