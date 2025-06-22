"""Convert between PrettyMIDI and Numba MIDI Score objects."""

import numpy as np
import pretty_midi

from numba_midi.score import (
    ControlArray,
    get_pedals_from_controls,
    NoteArray,
    PitchBendArray,
    Score,
    SignatureArray,
    TempoArray,
    Track,
)


def from_pretty_midi(midi: pretty_midi.PrettyMIDI) -> Score:
    """Convert a PrettyMIDI object to a Score object."""
    tracks = []
    for _, instrument in enumerate(midi.instruments):
        notes = NoteArray.zeros(len(instrument.notes))
        for note_id, note in enumerate(instrument.notes):
            notes.start[note_id] = note.start
            start_tick = midi.time_to_tick(note.start)
            notes.start_tick[note_id] = start_tick
            notes.duration[note_id] = note.end - note.start
            end_tick = midi.time_to_tick(note.end)
            notes.duration_tick[note_id] = end_tick - start_tick
            notes.pitch[note_id] = note.pitch
            notes.velocity[note_id] = note.velocity

        pitch_bends = PitchBendArray.zeros(len(instrument.pitch_bends))
        for pitch_bend_id, pitch_bend in enumerate(instrument.pitch_bends):
            pitch_bends.time[pitch_bend_id] = pitch_bend.time
            pitch_bends.value[pitch_bend_id] = pitch_bend.pitch
            pitch_bends.tick[pitch_bend_id] = midi.time_to_tick(pitch_bend.time)

        controls = ControlArray.zeros(len(instrument.control_changes))
        for control_id, control in enumerate(instrument.control_changes):
            controls.time[control_id] = control.time
            controls.value[control_id] = control.value
            controls.tick[control_id] = midi.time_to_tick(control.time)
            controls.number[control_id] = control.number

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

    ticks_per_quarter = midi.resolution
    tempo_change_times, tempi = midi.get_tempo_changes()
    tempo = TempoArray.zeros(len(tempo_change_times))

    clocks_per_click = 24  # Looks like we don't have this information in pretty_midi
    notated_32nd_notes_per_beat = 8  # Looks like we don't have this information in pretty_midi

    tempo.time = tempo_change_times
    tempo.quarter_notes_per_minute = tempi
    # 60.0/(midi._tick_scales[0][1]*midi.resolution)
    tempo.tick = np.array([midi.time_to_tick(t) for t in tempo_change_times])

    time_signatures = SignatureArray(
        numerator=[event.numerator for event in midi.time_signature_changes],
        denominator=[event.denominator for event in midi.time_signature_changes],
        tick=[midi.time_to_tick(event.time) for event in midi.time_signature_changes],
        time=[event.time for event in midi.time_signature_changes],
        clocks_per_click=[clocks_per_click],
        notated_32nd_notes_per_beat=[notated_32nd_notes_per_beat],
    )
    end_time = midi.get_end_time()
    last_tick = midi.time_to_tick(end_time)
    score = Score(
        tracks=tracks,
        last_tick=last_tick,
        time_signature=time_signatures,
        tempo=tempo,
        ticks_per_quarter=ticks_per_quarter,
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
        tick_scale = 60.0 / (tempo.quarter_notes_per_minute * midi.resolution)
        midi._tick_scales.append((int(tempo.tick), tick_scale))
    # Create list that maps ticks to time in seconds
    midi._update_tick_to_time(score.last_tick)

    for track in score.tracks:
        instrument = pretty_midi.Instrument(
            program=track.program,
            is_drum=track.is_drum,
            name=track.name,
        )
        for note in track.notes:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.start + note.duration,
                )
            )
        for control in track.controls:
            instrument.control_changes.append(
                pretty_midi.ControlChange(
                    number=control.number,
                    value=control.value,
                    time=control.time,
                )
            )
        for pitch_bend in track.pitch_bends:
            instrument.pitch_bends.append(
                pretty_midi.PitchBend(
                    pitch=pitch_bend.value,
                    time=pitch_bend.time,
                )
            )
        midi.instruments.append(instrument)
    # Set the time signature
    for time_signature in score.time_signature:
        midi.time_signature_changes.append(
            pretty_midi.TimeSignature(
                numerator=int(time_signature.numerator),
                denominator=int(time_signature.denominator),
                time=float(time_signature.time),
            )
        )

    return midi
