"""Convert between PrettyMIDI and Numba MIDI Score objects."""

import numpy as np
import pretty_midi

from numba_midi.score import (
    control_dtype,
    get_pedals_from_controls,
    note_dtype,
    pitch_bend_dtype,
    Score,
    tempo_dtype,
    Track,
)


def from_pretty_midi(midi: pretty_midi.PrettyMIDI) -> Score:
    """Convert a PrettyMIDI object to a Score object."""
    tracks = []
    for _, instrument in enumerate(midi.instruments):
        notes = np.empty(len(instrument.notes), dtype=note_dtype)
        for note_id, note in enumerate(instrument.notes):
            notes[note_id]["start"] = note.start
            start_tick = midi.time_to_tick(note.start)
            notes[note_id]["start_tick"] = start_tick
            notes[note_id]["duration"] = note.end - note.start
            end_tick = midi.time_to_tick(note.end)
            notes[note_id]["duration_tick"] = end_tick - start_tick
            notes[note_id]["pitch"] = note.pitch
            notes[note_id]["velocity_on"] = note.velocity

        pitch_bends = np.empty(len(instrument.pitch_bends), dtype=pitch_bend_dtype)
        for pitch_bend_id, pitch_bend in enumerate(instrument.pitch_bends):
            pitch_bends[pitch_bend_id]["time"] = pitch_bend.time
            pitch_bends[pitch_bend_id]["value"] = pitch_bend.pitch
            pitch_bends[pitch_bend_id]["tick"] = midi.time_to_tick(pitch_bend.time)

        controls = np.empty(len(instrument.control_changes), dtype=control_dtype)
        for control_id, control in enumerate(instrument.control_changes):
            controls[control_id]["time"] = control.time
            controls[control_id]["value"] = control.value
            controls[control_id]["tick"] = midi.time_to_tick(control.time)
            controls[control_id]["number"] = control.number

        pedals = get_pedals_from_controls(controls)
        track = Track(
            name=instrument.name,
            notes=notes,
            program=instrument.program,
            is_drum=instrument.is_drum,
            channel=None,  # TODO: set this to the correct value
            midi_track_id=None,  # TODO: set this to the correct value
            controls=controls,
            pedals=pedals,
            pitch_bends=pitch_bends,
        )
        assert len(midi.time_signature_changes) <= 1, "Only one time signature change is supported"

        tracks.append(track)
    if len(midi.time_signature_changes) == 1:
        numerator = midi.time_signature_changes[0].numerator
        denominator = midi.time_signature_changes[0].denominator
    else:
        # default to 4/4 if no time signature changes are found
        numerator = 4
        denominator = 4
    ticks_per_quarter = midi.resolution
    tempo_change_times, tempi = midi.get_tempo_changes()
    tempo = np.empty(len(tempo_change_times), dtype=tempo_dtype)

    clocks_per_click = 0  # Looks like we don't have this information in pretty_midi
    notated_32nd_notes_per_beat = 0  # Looks like we don't have this information in pretty_midi

    tempo["time"] = tempo_change_times
    tempo["bpm"] = tempi
    # 60.0/(midi._tick_scales[0][1]*midi.resolution)
    tempo["tick"] = [midi.time_to_tick(t) for t in tempo_change_times]

    score = Score(
        tracks=tracks,
        duration=midi.get_end_time(),
        time_signature=(numerator, denominator),
        tempo=tempo,
        clocks_per_click=clocks_per_click,
        ticks_per_quarter=ticks_per_quarter,
        notated_32nd_notes_per_beat=notated_32nd_notes_per_beat,
    )
    return score


def to_pretty_midi(score: Score) -> pretty_midi.PrettyMIDI:
    """Convert a Score object to a PrettyMIDI object."""
    midi = pretty_midi.PrettyMIDI()
    midi.resolution = score.ticks_per_quarter
    # Set the tempo
    midi._tick_scales = []  # reset the tick scales
    for tempo in score.tempo:
        # look like pretty_midi does not have a way to set the tempo changes
        # through its exposed API
        tick_scale = 60.0 / (tempo["bpm"] * midi.resolution)
        midi._tick_scales.append((int(tempo["tick"]), tick_scale))
    # Create list that maps ticks to time in seconds
    midi._update_tick_to_time(score.last_tick())

    for track in score.tracks:
        instrument = pretty_midi.Instrument(
            program=track.program,
            is_drum=track.is_drum,
            name=track.name,
        )
        for note in track.notes:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=note["velocity_on"],
                    pitch=note["pitch"],
                    start=note["start"],
                    end=note["start"] + note["duration"],
                )
            )
        for control in track.controls:
            instrument.control_changes.append(
                pretty_midi.ControlChange(
                    number=control["number"],
                    value=control["value"],
                    time=control["time"],
                )
            )
        for pitch_bend in track.pitch_bends:
            instrument.pitch_bends.append(
                pretty_midi.PitchBend(
                    pitch=pitch_bend["value"],
                    time=pitch_bend["time"],
                )
            )
        midi.instruments.append(instrument)
    # Set the time signature
    midi.time_signature_changes.append(
        pretty_midi.TimeSignature(
            numerator=score.time_signature[0],
            denominator=score.time_signature[1],
            time=0,
        )
    )

    return midi
