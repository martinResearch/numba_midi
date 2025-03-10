# numba_midi
numba-accelerated python midi score processing library

# Design choices

The implementation is in pure python and as a results is portable and easy to modify.
We use numpy vectorized operations when possible and use numba for some operations that are hard to vectorize.
Using lexicographic sorting allows to perform many operations efficiently without python loops.

# Music score Interfaces

The `Score` class represent music score with notes prepresented as atomic items with start time and durations instead of note-on and note-off events that need to be matched. This representation is more convenient for offline processing. 

We also provide a `MidiScore` class is intended to be as close as possible to the raw midi data using midi events representations with note on and note off events. This class is used as an intermediate representation when reading and writing from/to midi files.


# Piano roll

We provide a `PianoRoll` dataclass with conversion function from and to midi scores.

# To Do

* support Midi writing
 
