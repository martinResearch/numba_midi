"""Example of loading a MIDI file and converting it to a pianoroll using numba_midi."""

from pathlib import Path

from numba_midi import load_score

midi_file = str(Path(__file__).parent / "data" / "numba_midi" / "2c6e8007babc7ee877f1d2130b6459af.mid")
score = load_score(midi_file, notes_mode="no_overlap")
print(score)

# Score(num_tracks=15, num_notes=6006, duration=214.118)
pianoroll = score.to_pianoroll(time_step=0.01, pitch_min=0, pitch_max=127, num_bin_per_semitone=1)
print(pianoroll)
