"""TinySoundFont interop module."""

import tinysoundfont
import tinysoundfont.midi

from numba_midi.score import attribute_midi_channels, Score


def to_tinysoundfont(score: Score) -> list[tinysoundfont.midi.Event]:
    """Convert a Score object to a list of TinySoundFont MIDI events."""
    mapping_channels = attribute_midi_channels(score, max_channels=0, use_multiple_tracks=False)
    # convert to events
    midi_events = []
    for track_id, track in enumerate(score.tracks):
        channel = mapping_channels[track_id]
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
