"""Regression tests for the drawing functionality.

This module contains tests that compare the output of the drawing functions
with last known good (LKG) images stored in the repository.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from numba_midi.midi_draw import (
    ColorTheme,
    draw_controls,
    draw_pianoroll,
    GridOptions,
    hex_to_rgb,
    PianorollBox,
    TrackColors,
)
from numba_midi.numba_draw import NumPyCanvas
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

# Directory to store the LKG (Last Known Good) images
LKG_DIR = Path(__file__).parent / "data" / "midi_draw"
os.makedirs(LKG_DIR, exist_ok=True)


def create_score_with_notes() -> Score:
    """Create a simple score with a few notes."""
    tempo = TempoArray(time=[0], tick=[0], quarter_notes_per_minute=[120])
    time_signature = SignatureArray(
        time=[0], tick=[0], numerator=[4], denominator=[4], clocks_per_click=[24], notated_32nd_notes_per_beat=[8]
    )
    score = Score(tracks=[], tempo=tempo, last_tick=480 * 50, time_signature=time_signature)
    track = Track(
        name="Track 1",
        program=0,
        is_drum=False,
        notes=NoteArray.zeros((0)),
        controls=ControlArray.zeros((0)),
        pedals=PedalArray.zeros((0)),
        pitch_bends=PitchBendArray.zeros((0)),
    )
    score.add_track(track)
    score.add_notes(
        track_id=0,
        time=np.array([0, 1, 2, 0, 2], dtype=np.float32),
        duration=np.array([1, 2, 3, 1, 3], dtype=np.float32),
        pitch=np.array([60, 62, 64, 67, 69], dtype=np.float32),
        velocity=np.array([80, 90, 100, 70, 80], dtype=np.float32),
    )
    num_points = 1000
    time = np.linspace(0, score.duration, num_points)
    frequency = 0.10
    value = 10 + 90 * (np.cos(2 * np.pi * time * frequency) * 0.5 + 0.5)
    number = np.full((num_points), 5)
    score.add_controls(track_id=0, time=time, number=number, value=value)

    return score


def test_piano_roll_drawing(update_lkg: bool = False) -> None:
    """Test that drawing a piano roll produces the expected image."""
    # Create a simple score with a few notes

    score = create_score_with_notes()

    # Create MidiDraw instance with default color theme and NumPy backend
    canvas = NumPyCanvas(200, 400)
    color_theme = ColorTheme(
        piano_roll_background_color_odd=hex_to_rgb("#C0C0C0"),
        piano_roll_background_color_even=hex_to_rgb("#ffffff"),
        subdivision_lines_color=hex_to_rgb("#c7dbc9"),
        beat_lines_color=hex_to_rgb("#59AC5E"),
        bar_lines_color=hex_to_rgb("#127c3e"),
    )

    # Define the piano roll box (time and pitch range)
    box = PianorollBox(time_left=0, time_right=5, pitch_bottom=55, pitch_top=72)

    track_colors = TrackColors(
        fill_colors=np.array(
            [
                hex_to_rgb("#A83636"),  # Track 1 color
                hex_to_rgb("#00FF00"),  # Track 2 color
                hex_to_rgb("#0000FF"),  # Track 3 color
            ]
        ),
        edge_colors=np.array(
            [
                hex_to_rgb("#BEBEBE"),  # Track 1 color
                hex_to_rgb("#BEBEBE"),  # Track 2 color
                hex_to_rgb("#BEBEBE"),  # Track 3 color
            ]
        ),
        thickness=np.array([2, 2, 2]),
        alpha=np.array([1.0, 1.0, 1.0]),
    )
    grid_options = GridOptions(
        draw_beats=True,
        draw_pitches=True,
        draw_subdivisions=True,
    )

    draw_pianoroll(
        score=score,
        canvas=canvas,
        grid_options=grid_options,
        track_colors=track_colors,
        box=box,
        color_theme=color_theme,
    )
    result_image = canvas._image
    result_image = (result_image).astype(np.uint8)
    # Path to the LKG image
    lkg_path = LKG_DIR / "piano_roll_lkg.png"

    if update_lkg:
        # Save the current result as the new LKG
        # Convert from float32 to uint8 if needed
        # Save as PNG using numpy-based image libraries
        Image.fromarray(result_image).save(lkg_path)
        print(f"Updated LKG image at {lkg_path}")
    else:
        # Compare with the existing LKG
        if not lkg_path.exists():
            pytest.fail(f"LKG image does not exist at {lkg_path}. Run with update_lkg=True to create it.")

        expected_image = np.array(Image.open(lkg_path))

        # Check image dimensions
        assert result_image.shape == expected_image.shape, (
            f"Image shape mismatch: got {result_image.shape}, expected {expected_image.shape}"
        )

        # Compare pixel values
        # Allow for small differences due to potential platform-specific rendering differences
        max_diff = np.max(np.abs(result_image.astype(np.float32) - expected_image.astype(np.float32)))
        assert max_diff < 1.0, f"Image differs from LKG by {max_diff} pixel values"


def test_control_curve_drawing(update_lkg: bool = False) -> None:
    """Test that drawing a control curve produces the expected image."""
    # Create a simple score with control points

    score = create_score_with_notes()

    # Create MidiDraw instance with default color theme and NumPy backend
    canvas = NumPyCanvas(height=100, width=400)
    canvas.clear(hex_to_rgb("#EEEEEE"))

    # Define the control view box (time range)
    time_left = 0.0
    time_right = score.duration

    # Draw the control curve for controller 1 (mod wheel)

    draw_controls(
        canvas=canvas,
        score=score,
        track_number=0,
        control_number=5,
        time_left=time_left,
        time_right=time_right,
        color=hex_to_rgb("#000000"),
    )

    # Path to the LKG image
    lkg_path = LKG_DIR / "control_curve_lkg.png"
    result_image = canvas._image

    if update_lkg:
        # Save the current result as the new LKG
        image = Image.fromarray(result_image)
        image.save(lkg_path)
        print(f"Updated LKG image at {lkg_path}")
    else:
        # Compare with the existing LKG
        if not lkg_path.exists():
            pytest.fail(f"LKG image does not exist at {lkg_path}. Run with update_lkg=True to create it.")

        expected_image = np.array(Image.open(lkg_path))

        # Check image dimensions
        assert result_image.shape == expected_image.shape, (
            f"Image shape mismatch: got {result_image.shape}, expected {expected_image.shape}"
        )

        # Compare pixel values
        # Allow for small differences due to potential platform-specific rendering differences
        max_diff = np.max(np.abs(result_image.astype(np.float32) - expected_image.astype(np.float32)))
        assert max_diff < 1.0, f"Image differs from LKG by {max_diff} pixel values"


if __name__ == "__main__":
    # When run directly, create/update the LKG images
    test_piano_roll_drawing(update_lkg=False)
    test_control_curve_drawing(update_lkg=False)
