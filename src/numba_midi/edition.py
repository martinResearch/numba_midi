"""This module contains functions for editing MIDI scores."""

import numpy as np
from numba_midi.numba_2dengine import rectangles_segment_intersections

from numba_midi.score import Score


def remove_pitch_bend(score: Score, track_id: int, time: float) -> None:
    """Remove pitch bend events at a specific time for a track."""
    track = score.tracks[track_id]
    pitch_bends = track.pitch_bends
    # Find the pitch bend events at the specified time
    mask = pitch_bends.time == time
    # Remove the pitch bend events
    pitch_bends = pitch_bends[~mask]
    track.pitch_bends = pitch_bends


def remove_control(score: Score, track_id: int, time: float, control_number: int) -> None:
    """Remove control events at a specific time for a track."""
    track = score.tracks[track_id]
    controls = track.controls
    # Find the control events at the specified time and control number
    mask = (controls.time == time) & (controls.number == control_number)
    # Remove the control events
    controls = controls[~mask]
    track.controls = controls


def quantize_interval(score: Score, start: float, end: float, quantization_per_note: int) -> tuple[float, float]:
    """Quantize the interval defined by start and end to the nearest quarter note."""
    quarter_notes_start = score.time_to_quarter_note(start)
    subdivision = quantization_per_note
    quarter_notes_start = np.floor(quarter_notes_start * subdivision / 4) * (4 / subdivision)
    quarter_notes_end = score.time_to_quarter_note(end)
    quarter_notes_end = np.ceil(quarter_notes_end * subdivision / 4) * (4 / subdivision)
    quarter_notes_end = max(quarter_notes_start + (4 / subdivision), quarter_notes_end)
    time_start = score.quarter_note_to_time(quarter_notes_start)
    time_end = score.quarter_note_to_time(quarter_notes_end)
    return time_start, time_end


def find_notes_at_position(
    score: Score,
    track_ids: list[int],
    time: float,
    pitch: int,
    border_width_ratio: float = 0.2,
    max_border_width_sec: float = 0.1,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Find notes at the given position."""
    selection: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # Dictionary to store (track_id, note_index, side)
    for track_id in track_ids:
        track = score.tracks[track_id]
        if len(track.notes) == 0:
            continue

        notes = track.notes
        overlapping = (notes.pitch == pitch) & (notes.start <= time) & (time <= notes.end)
        indices = np.nonzero(overlapping)[0]
        overlapping_notes = notes[indices]
        border_width_sec = np.minimum(notes.duration * border_width_ratio, max_border_width_sec)
        side = -1 * (time < overlapping_notes.start + border_width_sec) + 1 * (
            time > overlapping_notes.end - border_width_sec
        )
        if len(overlapping_notes) > 0:
            selection[track_id] = (indices, side)

    return selection


def find_notes_in_segment(
    score: Score, track_ids: list[int], pitch1: int, pitch2: int, time_1: float, time_2: float
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Find notes in the segment defined by time_1, time_2 and pitch1, pitch2."""
    selection = {}
    for track_id in track_ids:
        track = score.tracks[track_id]
        if len(track.notes) == 0:
            continue

        notes = track.notes
        rectangles = np.column_stack(
            (
                notes.start,
                notes.pitch - 0.5,
                notes.end,
                notes.pitch + 0.5,
            )
        )

        overlapping = rectangles_segment_intersections(rectangles, np.array([[time_1, pitch1], [time_2, pitch2]]))
        indices = np.nonzero(overlapping)[0]
        side = np.zeros(len(indices), dtype=np.int8)  # Initialize side array
        if len(indices) > 0:
            selection[track_id] = (indices, side)

    return selection


def find_notes_in_rectangle(
    score: Score, track_ids: list[int], selection_rectangle: tuple[float, float, float, float], tol: float
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Find notes in a selection rectangle defined by (t1, p1, t2, p2) with a tolerance."""
    t1, p1, t2, p2 = selection_rectangle
    time_start = t1 - tol
    time_end = t2 + tol
    pitch_top = p2
    pitch_bottom = p1

    # Find notes in the selection rectangle
    selection: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # Array to store (track_id, note_index) tuples

    for track_id in track_ids:
        track = score.tracks[track_id]
        note_starts = track.notes.start
        note_ends = track.notes.end
        pitches = track.notes.pitch

        # Use numpy operations to find notes in the selection rectangle
        start_in_range = (note_starts <= time_end) & (note_ends >= time_start)
        end_in_range = (note_ends >= time_start) & (note_starts <= time_end)
        in_time_range = start_in_range | end_in_range
        in_pitch_range = (pitches >= pitch_bottom) & (pitches <= pitch_top)
        selected_indices = np.nonzero(in_time_range & in_pitch_range)[0]
        sides = np.zeros(len(selected_indices), dtype=bool)  # Initialize sides array
        # -1 for start only in selection, 1 for end, 0 for both
        sides = start_in_range[selected_indices] - end_in_range[selected_indices]
        # Check if the notes are at the start or end of the selection rectangle
        if len(selected_indices) > 0:
            selection[track_id] = (selected_indices, sides)
    return selection


def remove_selected_notes(score: Score, selected_notes: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
    """Remove selected notes from the score."""
    new_notes_init = {}
    selected_notes_sides = {}
    for track_id, (selected_indices, _) in selected_notes.items():
        track = score.tracks[track_id]
        new_notes_init[track_id] = track.notes[selected_indices]
        selected_notes_sides[track_id] = np.zeros((len(new_notes_init[track_id])), dtype=bool)
        # remove the selected notes from the track
        track.notes.delete(selected_indices)


def move_selected_notes(
    score: Score, selected_notes: dict[int, tuple[np.ndarray, np.ndarray]], pitch_delta: float, time_delta: float
) -> None:
    """Move selected notes by pitch_delta and time_delta."""
    # Update all selected notes

    assert selected_notes is not None, "No notes selected for moving."
    for track_id, (note_indices, sides) in selected_notes.items():
        track = score.tracks[track_id]
        notes = track.notes[note_indices]
        # Calculate new pitch and start time
        new_pitch = (notes.pitch + pitch_delta).astype(np.int32)
        new_pitch = np.clip(new_pitch, 0, 127)  # Clamp to valid MIDI pitch range

        new_start = notes.start + time_delta * (sides <= 0)
        new_start = np.clip(new_start, 0, score.duration)

        new_end = notes.end + time_delta * (sides >= 0)
        new_end = np.clip(new_end, 0, score.duration)
        new_duration = new_end - new_start
        # remove notes with duration 0
        keep = new_duration > 0
        new_duration = new_duration[keep]
        new_start = new_start[keep]
        new_pitch = new_pitch[keep]
        velocity = notes.velocity[keep]
        # create new notes in the score
        new_notes_track = score.create_notes(
            start=new_start,
            pitch=new_pitch,
            duration=new_duration,
            velocity=velocity,
        )
        track.notes[note_indices] = new_notes_track
