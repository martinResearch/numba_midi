"""This module contains functions for editing MIDI scores."""

from typing import Iterable, Optional

import numpy as np

# from numba_midi.numba.engine2d import rectangles_segment_intersections
from numba_midi.cython.engine2d import rectangles_segment_intersections
from numba_midi.score import Notes, Score
from dataclasses import dataclass


@dataclass
class SelectedNote:
    track_id: int
    note_index: int
    side: int

@dataclass
class TrackNotesSelection:
    """Store information about selected notes in a track."""
    indices: np.ndarray  # note indices
    sides: np.ndarray  # -1 for start, 0 for both, 1 for end

@dataclass
class ScoreNotesSelection:
    """Store information about selected notes in a score."""
    track_selections: dict[int, TrackNotesSelection]

    @property
    def num_notes(self) -> int:
        return sum(selection.indices.size for selection in self.track_selections.values())

    def asdict_no_sides(self) -> dict[int, np.ndarray]:
        return {track_id: selection.indices for track_id, selection in self.track_selections.items()}

    def as_selected_notes(self) -> list[SelectedNote]:
        """Convert the selection to a list of SelectedNote."""
        selected_notes = []
        for track_id, selection in self.track_selections.items():
            for i in range(selection.indices.size):
                selected_notes.append(SelectedNote(
                    track_id=track_id,
                    note_index=selection.indices[i],
                    side=selection.sides[i]
                ))
        return selected_notes

    @classmethod
    def from_selected_notes(cls, selected_notes: list[SelectedNote]) -> "ScoreNotesSelection":
        """Convert a list of SelectedNote to a ScoreNotesSelection."""
        track_selections = {}
        for note in selected_notes:
            if note.track_id not in track_selections:
                track_selections[note.track_id] = TrackNotesSelection(
                    indices=np.array([], dtype=int),
                    sides=np.array([], dtype=int)
                )
            track_selections[note.track_id].indices = np.append(track_selections[note.track_id].indices, note.note_index)
            track_selections[note.track_id].sides = np.append(track_selections[note.track_id].sides, note.side)
        return cls(track_selections=track_selections)

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
    """Quantize the interval defined by start and end to the quantization."""
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
    track_ids: Iterable[int],
    time: float,
    pitch: int,
    border_width_ratio: float = 0.2,
    max_border_width_sec: float = 0.1,
) -> ScoreNotesSelection:
    """Find notes at the given position."""
    track_selections = {}
    for track_id in track_ids:
        track = score.tracks[track_id]
        if len(track.notes) == 0:
            continue

        notes = track.notes
        overlapping = (notes.pitch == pitch) & (notes.start <= time) & (time <= notes.end)
        indices = np.nonzero(overlapping)[0]
        overlapping_notes = notes[indices]
        border_width_sec = np.minimum(notes.duration[indices] * border_width_ratio, max_border_width_sec)
        side = -1 * (time < overlapping_notes.start + border_width_sec) + 1 * (
            time > overlapping_notes.end - border_width_sec
        )
        if len(overlapping_notes) > 0:
            track_selections[track_id] = TrackNotesSelection(indices=indices, sides=side)

    return ScoreNotesSelection(track_selections=track_selections)



def find_first_note_at_position(
    score: Score, track_ids: Iterable[int], time: float, pitch: int
) -> SelectedNote|None:
    """Find the first note at the given position."""
    selected_notes = find_notes_at_position(
        score=score,
        track_ids=track_ids,
        time=time,
        pitch=pitch,
    )
    # keep only the first note that matches the position
    if len(selected_notes.track_selections) > 0:
        return selected_notes.as_selected_notes()[0]
    return None



def find_notes_in_segment(
    score: Score, track_ids: Iterable[int], pitch1: int, pitch2: int, time_1: float, time_2: float
) -> ScoreNotesSelection:
    """Find notes in the segment defined by time_1, time_2 and pitch1, pitch2."""
    track_selections = {}
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
        sides = np.zeros(len(indices), dtype=np.int8)  # Initialize side array
        if len(indices) > 0:
            track_selections[track_id] = TrackNotesSelection(indices=indices, sides=sides)

    return ScoreNotesSelection(track_selections=track_selections)




def find_notes_in_rectangle(
    score: Score,
    track_ids: Iterable[int],
    selection_rectangle: tuple[float, float, float, float],
    tol: float,
    both_sides: bool = False,
) -> ScoreNotesSelection:
    """Find notes in a selection rectangle defined by (t1, p1, t2, p2) with a tolerance."""
    t1, p1, t2, p2 = selection_rectangle
    time_start = t1 - tol
    time_end = t2 + tol
    pitch_top = p2
    pitch_bottom = p1

    # Find notes in the selection rectangle
    track_selections: dict[str, TrackNotesSelection] = {}

    for track_id in track_ids:
        track = score.tracks[track_id]
        note_starts = track.notes.start
        note_ends = track.notes.end
        pitches = track.notes.pitch

        # Use numpy operations to find notes in the selection rectangle
        start_in_range = (note_starts <= time_end) & (note_starts >= time_start)
        end_in_range = (note_ends >= time_start) & (note_ends <= time_end)
        if both_sides:
            in_time_range = start_in_range & end_in_range
        else:
            in_time_range = start_in_range | end_in_range
        in_pitch_range = (pitches >= pitch_bottom) & (pitches <= pitch_top)
        selected_indices = np.nonzero(in_time_range & in_pitch_range)[0]
        num_selected = len(selected_indices)
        if num_selected == 0:
            continue

        # -1 for start only in selection, 1 for end, 0 for both
        sides = np.where(start_in_range[selected_indices], np.where(end_in_range[selected_indices], 0, 1), -1)

        # Create NotesSelection object for this track
        track_selections[track_id] = TrackNotesSelection(indices=selected_indices, sides=sides)

    return ScoreNotesSelection(track_selections=track_selections)


def remove_selected_notes(score: Score, selected_notes: ScoreNotesSelection) -> None:
    """Remove selected notes from the score."""
    new_notes_init = {}
    selected_notes_sides = {}
    for track_id, track_selection in selected_notes.track_selections.items():
        track = score.tracks[track_id]
        new_notes_init[track_id] = track.notes[track_selection.indices]
        selected_notes_sides[track_id] = np.zeros((len(new_notes_init[track_id])), dtype=bool)
        # remove the selected notes from the track
        track.notes.delete(track_selection.indices)


def remove_notes_in_segment(
    score: Score, track_ids: Iterable[int], pitch1: int, pitch2: int, time_1: float, time_2: float
) -> bool:
    """Remove notes in the segment defined by time_1, time_2 and pitch1, pitch2."""
    selected_notes = find_notes_in_segment(score, track_ids, pitch1, pitch2, time_1, time_2)
    if selected_notes:
        remove_selected_notes(score, selected_notes)
        return True
    return False


def move_selected_notes(
    score: Score,
    selected_notes: ScoreNotesSelection,
    pitch_delta: float,
    time_delta: float,
    ref_score: Optional[Score] = None,
) -> None:
    """Move selected notes by pitch_delta and time_delta."""
    # Update all selected notes
    if ref_score is None:
        ref_score = score

    assert selected_notes is not None, "No notes selected for moving."
    for track_id, track_selection in selected_notes.track_selections.items():
        track = ref_score.tracks[track_id]
        notes = track.notes[track_selection.indices]
        # Calculate new pitch and start time
        new_pitch = (notes.pitch + pitch_delta).astype(np.int32)
        new_pitch = np.clip(new_pitch, 0, 127)  # Clamp to valid MIDI pitch range

        new_start = notes.start + time_delta * (track_selection.sides <= 0)
        new_start = np.clip(new_start, 0, score.duration)

        new_end = notes.end + time_delta * (track_selection.sides >= 0)
        new_end = np.clip(new_end, 0, score.duration)
        new_duration = new_end - new_start
        # remove notes with duration 0
        keep = new_duration > 0
        new_duration = new_duration[keep]
        new_start = new_start[keep]
        new_pitch = new_pitch[keep]
        velocity = notes.velocity[keep]
        # create temporary new notes
        new_notes_track = score.create_notes(
            start=new_start,
            pitch=new_pitch,
            duration=new_duration,
            velocity=velocity,
        )
        # overwrite the selected notes in the track
        score.tracks[track_id].notes[track_selection.indices] = new_notes_track


def copy_selected_notes(score: Score, selected_notes: ScoreNotesSelection) -> dict[int, Notes]:
    """Copy selected notes from the score."""
    new_notes = {}
    for track_id, track_selection in selected_notes.track_selections.items():
        track = score.tracks[track_id]
        new_notes[track_id] = track.notes[track_selection.indices]
    return new_notes


def paste_notes(
    score: Score,
    notes: dict[int, Notes],
    time_offset: float = 0.0,
    pitch_offset: int = 0,
) -> ScoreNotesSelection:
    """Paste notes into the score with an optional time and pitch offset.
    return the new notes selection.
    """
    track_selections = {}
    for track_id, track_notes in notes.items():
        num_score_notes = len(score.tracks[track_id].notes)
        score.add_notes(
            track_id,
            time=track_notes.start + time_offset,
            duration=track_notes.duration,
            pitch=track_notes.pitch + pitch_offset,
            velocity=track_notes.velocity,
        )
        indices = np.arange(num_score_notes, num_score_notes + len(track_notes))
        sides = np.zeros(len(track_notes), dtype=np.int8)
        track_selections[track_id] = TrackNotesSelection(indices=indices, sides=sides)

    return ScoreNotesSelection(track_selections=track_selections)


def edit_notes(
    score: Score,
    note_indices: np.ndarray,
    track_id: int,
    start: np.ndarray,
    duration: np.ndarray,
    pitch: np.ndarray,
    velocity: np.ndarray,
) -> None:
    """Edit notes in the score."""
    track = score.tracks[track_id]

    new_notes = score.create_notes(
        start=start,
        pitch=pitch,
        duration=duration,
        velocity=velocity,
    )
    assert len(new_notes) == len(note_indices), "New notes length must match note indices length"
    track.notes[note_indices] = new_notes
