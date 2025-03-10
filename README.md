# numba_midi
A Numba-accelerated Python library for reading MIDI file and score processing.

### Installation 


```
pip install git+https://github.com/martinResearch/numba_midi.git
```


### Design Choices

The implementation is in pure Python, making it portable and easy to modify. We use NumPy vectorized operations when possible and leverage Numba for operations that are difficult to vectorize. Lexicographic sorting of NumPy arrays often allows for efficient operations without Python loops.

### Music Score Interfaces

The `Score` class represents a music score with notes as atomic items, including start times and durations, instead of note-on and note-off events that need to be matched. This representation is more convenient for offline processing.

The `MidiScore` class is designed to closely mirror raw MIDI data using MIDI events with note-on and note-off events. This class serves as an intermediate representation when reading from and writing to MIDI files.

### Piano Roll

We provide a `PianoRoll` dataclass with conversion functions to and from MIDI scores.

### To Do

* Support MIDI writing

### Alternatives

* [pypianoroll](https://github.com/salu133445/pypianoroll): Focused on piano roll functionalities. It is implemented with Python loops over notes and thus likely to be slow. It uses pretty-midi to load MIDI files, which is also slow.
* [symusic](https://github.com/Yikai-Liao/symusic): Implemented in C++ and interfaced with PyBind11. It is very fast, but the fact that it is implemented in C++ makes it harder to extend than pure Python. `numba_midi` is a bit slower but implemented in pure Python.