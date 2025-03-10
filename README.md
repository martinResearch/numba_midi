# numba_midi
A numba-accelerated Python library for MIDI score processing.

### Design Choices

The implementation is in pure Python, making it portable and easy to modify. We use NumPy vectorized operations when possible and leverage Numba for operations that are difficult to vectorize. Lexicographic sorting allows for efficient operations without Python loops.

### Music Score Interfaces

The `Score` class represents a music score with notes as atomic items, including start times and durations, instead of note-on and note-off events that need to be matched. This representation is more convenient for offline processing.

The `MidiScore` class is designed to closely mirror raw MIDI data using MIDI event representations with note-on and note-off events. This class serves as an intermediate representation when reading from and writing to MIDI files.

### Piano Roll

We provide a `PianoRoll` dataclass with conversion functions to and from MIDI scores.

### To Do

* Support MIDI writing
