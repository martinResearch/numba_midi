"""Conversion functions from and to symusic score objects."""

import numpy as np
import symusic
import symusic.types

from numba_midi.score import (
    ControlArray,
    NoteArray,
    PedalArray,
    PitchBendArray,
    Score,
    SignatureArray,
    TempoArray,
    Track,
)


def from_symusic(symusic_score: symusic.types.Score) -> Score:
    """Convert a symusic object to a Numba-compatible format."""
    tracks = []
    score_ticks = symusic_score.to(symusic.TimeUnit.tick)
    score_seconds = symusic_score.to(symusic.TimeUnit.second)

    for _, (track_ticks, track_secs) in enumerate(zip(score_ticks.tracks, score_seconds.tracks, strict=False)):
        notes = NoteArray.zeros(len(track_ticks.notes))
        track_ticks_notes_numpy = track_ticks.notes.numpy()
        track_secs_notes_numpy = track_secs.notes.numpy()
        notes.start = track_secs_notes_numpy["time"]
        notes.start_tick = track_ticks_notes_numpy["time"]
        notes.duration_tick = track_ticks_notes_numpy["duration"]
        notes.duration = track_secs_notes_numpy["duration"]
        notes.pitch = track_secs_notes_numpy["pitch"]
        notes.velocity = track_secs_notes_numpy["velocity"]

        controls = ControlArray.zeros(len(track_ticks.controls))

        control_tick_numpy = track_ticks.controls.numpy()
        control_secs_numpy = track_secs.controls.numpy()
        controls.time = control_secs_numpy["time"]
        controls.value = control_tick_numpy["value"]
        controls.tick = control_tick_numpy["time"]
        controls.number = control_tick_numpy["number"]

        pitch_bends = PitchBendArray.zeros(len(track_ticks.pitch_bends))
        pitch_bends_tick_numpy = track_ticks.pitch_bends.numpy()
        pitch_bends_secs_numpy = track_secs.pitch_bends.numpy()
        pitch_bends.time = pitch_bends_secs_numpy["time"]
        pitch_bends.value = pitch_bends_secs_numpy["value"]
        pitch_bends.tick = pitch_bends_tick_numpy["time"]

        pedals = PedalArray.zeros(len(track_ticks.pedals))
        pedal_tick_numpy = track_ticks.pedals.numpy()
        pedal_secs_numpy = track_secs.pedals.numpy()
        pedals.time = pedal_secs_numpy["time"]
        pedals.duration = pedal_secs_numpy["duration"]
        pedals.tick = pedal_tick_numpy["time"]
        pedals.duration_tick = pedal_tick_numpy["duration"]
        track = Track(
            name=track_ticks.name,
            notes=notes,
            program=track_ticks.program,
            is_drum=track_ticks.is_drum,
            channel=None,
            midi_track_id=None,
            controls=controls,
            pedals=pedals,
            pitch_bends=pitch_bends,
        )

        tracks.append(track)

    assert len(symusic_score.time_signatures) <= 1, "Only one time signature change is supported"
    if len(symusic_score.time_signatures) == 1:
        numerator = symusic_score.time_signatures[0].numerator
        denominator = symusic_score.time_signatures[0].denominator
    else:
        # default to 4/4 if no time signature changes are found
        numerator = 4
        denominator = 4

    tempos_sec_numpy = score_seconds.tempos.numpy()
    tempo_tick_numpy = score_ticks.tempos.numpy()
    tempo = TempoArray.zeros(len(score_seconds.tempos))
    tempo.time = tempos_sec_numpy["time"]
    tempo.quarter_notes_per_minute = 60000000 / tempo_tick_numpy["mspq"]
    tempo.tick = tempo_tick_numpy["time"]

    # Note sure where to find this in the symusic object
    clocks_per_click = 0
    notated_32nd_notes_per_beat = 0

    time_signature = SignatureArray(
        time=[0],
        tick=[0],
        numerator=[numerator],
        denominator=[denominator],
        clocks_per_click=[clocks_per_click],
        notated_32nd_notes_per_beat=[notated_32nd_notes_per_beat],
    )
    score = Score(
        tracks=tracks,
        last_tick=score_ticks.end(),
        tempo=tempo,
        time_signature=time_signature,
        ticks_per_quarter=symusic_score.ticks_per_quarter,
    )

    return score


def to_symusic(score: Score) -> symusic.types.Score:
    """Convert a Numba-compatible score to a symusic object."""
    tracks = []

    tempo = symusic.Tempo.from_numpy(
        time=score.tempo.tick.astype(np.int32), mspq=(60000000 / score.tempo.quarter_notes_per_minute).astype(np.int32)
    )

    for track in score.tracks:
        controls = symusic.ControlChange.from_numpy(
            number=track.controls.number, time=track.controls.tick, value=track.controls.value
        )
        pedals = symusic.Pedal.from_numpy(time=track.pedals.tick, duration=track.pedals.duration_tick)
        pitch_bends = symusic.PitchBend.from_numpy(time=track.pitch_bends.tick, value=track.pitch_bends.value)
        notes = symusic.Note.from_numpy(
            pitch=track.notes.pitch.astype(np.int8),
            velocity=track.notes.velocity.astype(np.int8),
            time=track.notes.start_tick.astype(np.int32),
            duration=track.notes.duration_tick.astype(np.int32),
        )

        symusic_track = symusic.Track(
            name=track.name,
            program=track.program,
            is_drum=track.is_drum,
            notes=notes,
            controls=controls,
            pedals=pedals,
            pitch_bends=pitch_bends,
        )
        tracks.append(symusic_track)

    symusic_score = symusic.Score()
    symusic_score.ticks_per_quarter = score.ticks_per_quarter
    # TODO : Add support for multiple time signatures
    symusic_score.time_signatures = symusic.TimeSignature.from_numpy(
        time=score.time_signature.tick.astype(np.int32),
        numerator=score.time_signature.numerator.astype(np.int8),
        denominator=score.time_signature.denominator.astype(np.int8),
    )

    symusic_score.tempos = tempo
    symusic_score.tracks.extend(tracks)
    return symusic_score
