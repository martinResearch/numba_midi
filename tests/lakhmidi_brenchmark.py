

import requests
import tarfile
from pathlib import Path
import glob
import tqdm
from numba_midi.score import load_score
import time
import pandas as pd


def decompress_tar_gz_with_progress(file_path, output_dir):
    with tarfile.open(file_path, "r:gz") as tar:
        members = tar.getmembers()
        total_files = len(members)        
        with tqdm.tqdm(total=total_files, desc="Decompressing", unit="file") as pbar:
            for member in members:
                tar.extract(member, path=output_dir)
                pbar.update(1)

def download_dataset():
    folder = Path(__file__).parent / "data" / "lakh"
    folder.mkdir(parents=True, exist_ok=True)
    url = "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz"
    print (f"Downloading Lakh MIDI Dataset from {url}...")
    file = folder/"lmd_matched.tar.gz"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def decompress_dataset():
    folder = Path(__file__).parent / "data" / "lakh"
    file = folder/"lmd_matched.tar.gz"  
    # decompress with progress bar
    decompress_tar_gz_with_progress(file, str(folder))


def benchmark():
    # get the list of midi files
    midi_files = glob.glob(
        str(Path(r"C:\repos\audio_to_midi\src\audio_to_midi\datasets\lakh_midi\lmd_matched") / "**" / "*.mid"),
        recursive=True,
    )
    print("Number of MIDI files:", len(midi_files))
    duration_numba_midi={}
    failures_numba_midi=set()

    midi_files=midi_files[:100]
    print("Benchmarking numba_midi...")
    for midi_file in tqdm.tqdm(midi_files):       
        try:
             # load row midi score
            start_time = time.perf_counter()
            score = load_score(midi_file)
            duration_numba_midi[midi_file] = time.perf_counter() - start_time
        except Exception as e:
            failures_numba_midi.add(midi_file)


    df = pd.DataFrame(duration_numba_midi)

    print("Benchmarking symusic...")
    for midi_file in tqdm.tqdm(midi_files):       
        try:
             # load row midi score
            start_time = time.perf_counter()
            score = load_score(midi_file)
            duration_numba_midi[midi_file] = time.perf_counter() - start_time
        except Exception as e:
            failures_numba_midi.add(midi_file)


if __name__ == "__main__":
    #download_dataset()
    decompress_dataset()
    benchmark()