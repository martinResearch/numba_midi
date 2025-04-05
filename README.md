# numba_midi


![python_package](https://github.com/martinResearch/numba_midi/actions/workflows//python-package.yml/badge.svg)

A Numba-accelerated Python library for fast MIDI file reading and music score processing.

This library is implemented entirely in Python, making it portable and easy to modify and extend for python developpers. Efficiency is achieved by using NumPy structured arrays to store data instead of creating per-event or per-note Python class instances. The library leverages NumPy vectorized operations where possible and uses Numba for non-vectorizable operations. 

## Main features

* read and write midi files
* pure python, making its internals more accessible to python developpers
* 10x faster than *pretty_midi* for reading
* events (note on and off) and notes (start and duration) representations
* tracks representation based on numpy arrays making it trivial to do vectorized operations on all notes in a track.
* multiple modes regarding how to process overlapping notes when converting from events to note representation. 
* conversion to and from priano roll representation
* conversion functions from/to pretty_midi and symusic
* timestamps and durations both in seconds and ticks

## Installation

To install the library, use the following command:

```bash
pip install git+https://github.com/martinResearch/numba_midi.git
```

## Music Score Interfaces

- **`Score`**: Represents a music score with notes as atomic items, including start times and durations. This approach is more convenient for offline processing compared to handling note-on and note-off events.
- **`MidiScore`**: Mirrors raw MIDI data by using MIDI events, including note-on and note-off events. This class serves as an intermediate representation for reading and writing MIDI files.

## Piano Roll

The library includes a `PianoRoll` dataclass with conversion functions to seamlessly transform between piano rolls and MIDI scores.

## Interopability

We provide functions to convert from/to score from the **symusic** and **pretty_midi** liberaries in 
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

We provide control to the user on how to handle overlapping note and zero length notes
through the parameter `notes_mode` with type `NotesMode = Literal["no_overlap", "first_in_first_out", "note_off_stops_all"]`

We obtain the same behavior as *pretty-midi* when using `notes_mode="note_off_stops_all"` and the same behavior as *symusic* when using `notes_mode="first_in_first_out"`.

**Note:** Using `"no overlap"` is not as strong as enforcing a monophonic constraint on the instrument: two notes with different pitches can still overlap in time. Although polyphonic, a piano should use `"no overlap"`to be realistic.

## Alternatives

Here are some alternative libraries and how they compare to `numba_midi`:
- **[pretty_midi](https://craffel.github.io/pretty-midi/)**. Implemented using a python object for each note, making it slow compared to `numpa_midi`.
- **[pypianoroll](https://github.com/salu133445/pypianoroll)**: Focused on piano roll functionalities. It relies on Python loops over notes, which can be slow. It also uses `pretty-midi` for MIDI file loading, which is not optimized for speed.
- **[symusic](https://github.com/Yikai-Liao/symusic)**: Written in C++ and interfaced with PyBind11, making it extremely fast. However, its C++ implementation makes it much harder to extend for python developpers compared to a pure Python libraries like `numba_midi`.
- **[muspy](https://github.com/salu133445/muspy)**: Represents music scores using Python classes, with one `Note` class instance per note. This design prevents the use of efficient NumPy vectorized operations, relying instead on slower Python loops.

