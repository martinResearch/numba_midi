# numba_midi
A Numba-accelerated Python library for fast MIDI file reading and score processing.

The implementation is in pure Python, making it portable and easy to modify. We made the code efficient by using only few NumPy structured arrays to store the data, using NumPy vectorized operations when possible and leveraging Numba for operations that cannot be vectorized. Using lexicographic sorting on NumPy arrays often allows for efficient operations without large Python loops.


### Installation 

```
pip install git+https://github.com/martinResearch/numba_midi.git
```

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
* [muspy](https://github.com/salu133445/muspy): represent the music score using python classes with one `Note` class instance for each note. As a result, one cannot use efficient NumPy vectorized operations and must use python loops, which make the processing slower.
