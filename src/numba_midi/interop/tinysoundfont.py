"""TinySoundFont interop module."""

import tinysoundfont

from numba_midi.score import merge_non_overlapping_tracks, reattribute_midi_channels, Score


def to_tinysoundfont(score: Score) -> list[tinysoundfont.midi.Event]:
    """Convert a Score object to a list of TinySoundFont MIDI events."""
    # remove empty tracks
    score_cleaned = score.without_empty_tracks()
    # merge non-overlapping tracks that use the same program to reduce the number of channels required
    score_cleaned = merge_non_overlapping_tracks(score_cleaned)
    # reattribute MIDI channels to tracks
    reattribute_midi_channels(score_cleaned)
    # conver to events
    midi_events = []
    for track in score_cleaned.tracks:
        channel = track.channel
        midi_events.append(
            tinysoundfont.midi.Event(
                action=tinysoundfont.midi.ProgramChange(track.program), t=0.0, channel=channel, persistent=True
            )
        )

        for note in track.notes:
            midi_events.append(
                tinysoundfont.midi.Event(
                    action=tinysoundfont.midi.NoteOn(int(note.pitch), int(note.velocity)),
                    t=float(note.start),
                    channel=channel,
                    persistent=True,
                )
            )
            note_end = note.start + note.duration
            midi_events.append(
                tinysoundfont.midi.Event(
                    action=tinysoundfont.midi.NoteOff(int(note.pitch)),
                    t=float(note_end),
                    channel=channel,
                    persistent=True,
                )
            )
        for control in track.controls:
            midi_events.append(
                tinysoundfont.midi.Event(
                    action=tinysoundfont.midi.ControlChange(int(control.number), int(control.value)),
                    t=float(control.time),
                    channel=channel,
                    persistent=True,
                )
            )
        for pitch_bend in track.pitch_bends:
            midi_events.append(
                tinysoundfont.midi.Event(
                    action=tinysoundfont.midi.PitchBend(int(pitch_bend.value) + 8192),
                    t=float(pitch_bend.time),
                    channel=channel,
                    persistent=True,
                )
            )
    midi_events.sort(key=lambda x: x.channel)
    midi_events.sort(key=lambda x: x.t)
    return midi_events
