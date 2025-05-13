# numba_midi

![python_package](https://github.com/martinResearch/numba_midi/actions/workflows//python-package.yml/badge.svg)

A Numba-accelerated Python library for fast MIDI file reading and music score processing.

This library is implemented entirely in Python, making it portable and easy to modify and extend for Python developers. Efficiency is achieved by using NumPy structured arrays to store data instead of creating per-event or per-note Python class instances. The library leverages NumPy vectorized operations where possible and uses Numba for non-vectorizable operations. The set of dependencies is minimal and limited to `numpy` and `numba`.

## Main features

* Read and write MIDI files.
* Pure Python, making its internals more accessible to Python developers.
* 7x faster on average than *pretty_midi* for reading a MIDI file from disk.
* Events (note on and off) and notes (start and duration) representations.
* Tracks representation based on NumPy arrays, making it trivial to do vectorized operations on all notes in a track.
* Multiple modes regarding how to process overlapping notes when converting from events to note representation.
* Conversion to and from piano roll representation.
* Conversion functions from/to `pretty_midi` and `symusic`.
* Timestamps and durations both in seconds and ticks.

## Installation

To install the library, use the following command:

```bash
pip install numba_midi
```

## Music Score Interfaces

- **`Score`**: Represents a music score with notes as atomic items, including start times and durations. This approach is more convenient for offline processing compared to handling note-on and note-off events.
- **`MidiScore`**: Mirrors raw MIDI data by using MIDI events, including note-on and note-off events. This class serves as an intermediate representation for reading and writing MIDI files.

## Piano Roll

The library includes a `PianoRoll` dataclass with conversion functions to seamlessly transform between piano rolls and MIDI scores.


## Example

example in [example.py](./tests/example.py):
```
from pathlib import Path
from numba_midi import load_score

midi_file = str(Path(__file__).parent / "data" / "numba_midi" / "2c6e8007babc7ee877f1d2130b6459af.mid")
score = load_score(midi_file, notes_mode="no_overlap")
print(score)
# Score(num_tracks=15, num_notes=6006, duration=214.118)

pianoroll = score.to_pianoroll(time_step=0.01, pitch_min=0, pitch_max=127, num_bin_per_semitone=1)
print(pianoroll)
```

## Interoperability

We provide functions to convert from/to score from the **symusic** and **pretty_midi** libraries in 
[symusic.py](./src/numba_midi/interop/symusic.py) 
and [pretty_midi.py](./src/numba_midi/interop/pretty_midi.py) respectively.

## Overlapping Notes Behavior

MIDI files can contain tracks with notes that overlap in channel, pitch, and time. How to convert these to notes with start times and durations depends on the chosen convention. Ideally, we want to choose the one that matches how the synthesizer will interpret the MIDI events.

For example, for a given channel and pitch, we can have:

| tick | channel | type | pitch | velocity |
|------|---------|------|-------|----------|
| 100  | 1       | On   | 80    | 60       |
| 110  | 1       | On   | 80    | 60       |
| 120  | 1       | On   | 80    | 60       |
| 120  | 1       | Off  | 80    | 0        |
| 130  | 1       | Off  | 80    | 0        |
| 140  | 1       | Off  | 80    | 0        |
| 150  | 1       | On   | 80    | 60       |
| 150  | 1       | Off  | 80    | 0        |
| 160  | 1       | Off  | 80    | 0        |

Should the *Off* event on tick 120 stop all three notes, the first two notes, or just the first one?  
Should the first note stop at tick 110 when we have a new note to avoid any overlap?  
Should we create a note with duration 0 or 10 starting on tick 150, or no note at all?  
If a note is not closed when we reach the end of the song, should it be discarded, or should we keep it and use the end of the song as the end time?

We provide control to the user on how to handle overlapping notes and zero-length notes through the parameter `notes_mode` with type `NotesMode = Literal["no_overlap", "first_in_first_out", "note_off_stops_all"]`.

We obtain the same behavior as *pretty-midi* when using `notes_mode="note_off_stops_all"` and the same behavior as *symusic* when using `notes_mode="first_in_first_out"`.

**Note:** Using `"no_overlap"` is not as strong as enforcing a monophonic constraint on the instrument: two notes with different pitches can still overlap in time. Although polyphonic, a piano should use `"no_overlap"` to be realistic.

## Benchmark

We measure the loading speed by taking the first 1000 MIDI files (after sorting the paths) from the Lakh matched MIDI dataset. We measure both the time it takes to load from disk and the time it takes to load from raw bytes already in memory if that is available. We take the minimum duration over 10 runs for each file. We compute the loading speed in MB/sec for each file and compute the median values. We ignore the files that could not be loaded when computing the median. The benchmark was executed with Python 3.11.10 
on a laptop with an Intel i7-13800H processor clocked at 2.9 GHz.

| Library   | Disk Median MB/s | Disk Average MB/s | Memory Median MB/s | Memory Average MB/s | #Failures |
|-----------|------------------|-------------------|---------------------|---------------------|-----------|
| numba_midi | 3.8              | 4.1               | 5.8                 | 5.6                 | 4         |
| symusic    | 85               | 86                | 164                 | 172                 | 4         |
| pretty_midi| 0.52             | 0.52              | x                   | x                   | 8         |

When reading from disk, we are about 7x faster than *pretty_midi* and 22x slower than *symusic*.

Our library has also been benchmarked against alternatives on the [symusic page](https://github.com/Yikai-Liao/symusic). In this benchmark *numba-midi* load midi files 15x faster than *pretty_midi* and 20x slower than *symusic* when comparing the median speed. 

**Note:** We could probably get a 2x speedup with a reasonable amount of effort by moving more code to Numba JIT-compiled functions.

# TODO

* add conversion function from and to pypianoroll structure
* add conversion function from and to muspy structure
* add pianoroll plotting functions into numpy arrays and matplotlib plots.

## Alternatives

Here are some alternative libraries and how they compare to `numba_midi`:
- **[mido](https://github.com/mido/mido)**. Library for working with MIDI messages and ports. It allows to load midi file to event-based representation. It is implemented in pure python with one python class instance for each event and and is slow.
- **[pretty_midi](https://craffel.github.io/pretty-midi/)**: Implemented using a Python object for each note, making it slow compared to `numba_midi`.
- **[pypianoroll](https://github.com/salu133445/pypianoroll)**: Focused on piano roll functionalities. It relies on Python loops over notes, which can be slow. It also uses `pretty-midi` for MIDI file loading, which is not optimized for speed.
- **[symusic](https://github.com/Yikai-Liao/symusic)**: Written in C++ and interfaced with PyBind11, making it extremely fast. However, its C++ implementation makes it much harder to extend for Python developers compared to pure Python libraries like `numba_midi`.
- **[muspy](https://github.com/salu133445/muspy)**: Represents music scores using Python classes, with one `Note` class instance per note. This design prevents the use of efficient NumPy vectorized operations, relying instead on slower Python loops.
- **[partitura](https://github.com/CPJKU/partitura)**: Supports loading from and exporting to MusicXML and MIDI files. Uses mido under the hood which makes it slow.
