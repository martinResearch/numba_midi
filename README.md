# numba_midi
A Numba-accelerated Python library for fast MIDI file reading and music score processing.

This library is implemented entirely in Python, making it portable and easy to modify. Efficiency is achieved by using NumPy structured arrays to store data instead of creating per-event or per-note class instances. The library leverages NumPy vectorized operations where possible and uses Numba for non-vectorizable operations. Lexicographic sorting on NumPy arrays minimizes the need for large Python loops, enabling efficient execution.

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

## Alternatives

Here are some alternative libraries and how they compare to `numba_midi`:
- **[pretty_midi](https://craffel.github.io/pretty-midi/)**. Implemented using a python object for each note, making it slow compared to `numpa_midi`.
- **[pypianoroll](https://github.com/salu133445/pypianoroll)**: Focused on piano roll functionalities. It relies on Python loops over notes, which can be slow. It also uses `pretty-midi` for MIDI file loading, which is not optimized for speed.
- **[symusic](https://github.com/Yikai-Liao/symusic)**: Written in C++ and interfaced with PyBind11, making it extremely fast. However, its C++ implementation makes it less extensible compared to pure Python libraries like `numba_midi`.
- **[muspy](https://github.com/salu133445/muspy)**: Represents music scores using Python classes, with one `Note` class instance per note. This design prevents the use of efficient NumPy vectorized operations, relying instead on slower Python loops.

