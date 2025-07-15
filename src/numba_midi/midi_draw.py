"""Drawing Utility to draw pianroll, note velocities and control curves."""

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, Union

import numpy as np

from numba_midi.score import ControlArray, NoteArray, Score


@dataclass
class PianorollBox:
    """Box of time and pitch domain."""

    time_left: float
    time_right: float
    pitch_top: float
    pitch_bottom: float


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert a hex color string to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("Hex color must be 6 characters long")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


@dataclass
class TrackColors:
    """Colors for the tracks in the piano roll."""

    fill_colors: np.ndarray
    edge_colors: np.ndarray
    thickness: np.ndarray
    alpha: np.ndarray


@dataclass
class ColorTheme:
    """A class representing a color theme for the MIDI drawings."""

    piano_roll_background_color_odd: Tuple[int, int, int]
    piano_roll_background_color_even: Tuple[int, int, int]
    subdivision_lines_color: Tuple[int, int, int]
    beat_lines_color: Tuple[int, int, int]
    bar_lines_color: Tuple[int, int, int]


@dataclass
class GridOptions:
    """Options for the grid in the piano roll."""

    draw_beats: bool
    draw_pitches: bool
    draw_subdivisions: bool


@dataclass
class PixelMapping:
    """A class representing the mapping of pixels to time and pitch."""

    pixels_per_second: float
    pixels_per_pitch: float
    time_left: float
    pitch_top: float


def get_pixel_mapping(width: int, height: int, box: PianorollBox) -> PixelMapping:
    """Get the pixel mapping for the piano roll."""
    return PixelMapping(
        pixels_per_second=(width - 1) / (box.time_right - box.time_left),
        pixels_per_pitch=(height - 1) / (box.pitch_top - box.pitch_bottom),
        time_left=box.time_left,
        pitch_top=box.pitch_top,
    )


class CanvasProto(Protocol):
    """Abstract base class for drawing canvas.

    Implement this interface to create a new drawing backend for MidiDraw.
    """

    @property
    def height(self) -> int: ...

    @property
    def width(self) -> int: ...

    def draw_rectangles(
        self,
        rectangles: np.ndarray,
        fill_colors: Union[np.ndarray, Tuple[int, int, int]],
        alpha: Union[np.ndarray, float] = 1.0,
        edge_colors: Optional[Union[np.ndarray, Tuple[int, int, int]]] = None,
        thickness: Union[np.ndarray, int] = 0,
        fill_alpha: Optional[Union[np.ndarray, float]] = None,
        edge_alpha: Optional[Union[np.ndarray, float]] = None,
    ) -> None:
        """Draw rectangles on the surface.

        Args:
            surface: The surface to draw on.
            rectangles: Array of rectangle coordinates [x1, y1, x2, y2].
            fill_colors: Fill color(s) for the rectangles.
            alpha: Alpha/opacity value(s) for the rectangles.
            edge_colors: Edge color(s) for the rectangles.
            thickness: Thickness of the rectangle edges.
            fill_alpha: Alpha/opacity for the fill (overrides alpha).
            edge_alpha: Alpha/opacity for the edges (overrides alpha).
        """
        ...

    def draw_polyline(
        self,
        x: np.ndarray,
        y: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int = 1,
        alpha: float = 1.0,
    ) -> None:
        """Draw a polyline on the surface.

        Args:
            surface: The surface to draw on.
            x: X-coordinates of the polyline points.
            y: Y-coordinates of the polyline points.
            color: Color for the staircase line.
            thickness: Line thickness.
            alpha: Alpha/opacity for the line.
        """
        ...


def draw_pianoroll(
    canvas: CanvasProto,
    score: Score,
    track_colors: TrackColors,
    box: PianorollBox,
    grid_options: GridOptions,
    color_theme: ColorTheme,
) -> None:
    """Draw the piano roll on the canvas."""
    draw_piano_roll_background(canvas=canvas, score=score, box=box, grid_options=grid_options, color_theme=color_theme)
    draw_score_notes(canvas=canvas, box=box, track_colors=track_colors, score=score)


def draw_piano_roll_background(
    score: Score, canvas: CanvasProto, box: PianorollBox, grid_options: GridOptions, color_theme: ColorTheme
) -> None:
    # draw light grey rectangle every other pitch

    # Fill the background using draw_rectangles for the even color
    height = canvas.height
    width = canvas.width
    background_rectangle = np.array([[0, 0, height, width]], dtype=np.int32)
    background_color = color_theme.piano_roll_background_color_even
    canvas.draw_rectangles(
        background_rectangle,
        fill_colors=background_color,
        alpha=1.0,
        edge_colors=None,
        thickness=0,
    )

    pixel_mapping = get_pixel_mapping(width, height, box)
    if grid_options.draw_pitches:
        pitches_rectangles = np.column_stack(
            (
                np.zeros((127), dtype=np.int32),
                (box.pitch_bottom - np.arange(127) - 0.5) * pixel_mapping.pixels_per_pitch,
                np.full((127), width),
                (box.pitch_top - np.arange(127) + 0.5) * pixel_mapping.pixels_per_pitch,
            )
        ).astype(np.int32)

        pitches_fill_colors = np.empty((127, 3), dtype=np.uint8)
        pitches_fill_colors[1::2, 0] = color_theme.piano_roll_background_color_odd[0]
        pitches_fill_colors[1::2, 1] = color_theme.piano_roll_background_color_odd[1]
        pitches_fill_colors[1::2, 2] = color_theme.piano_roll_background_color_odd[2]
        pitches_fill_colors[::2, 0] = color_theme.piano_roll_background_color_even[0]
        pitches_fill_colors[::2, 1] = color_theme.piano_roll_background_color_even[1]
        pitches_fill_colors[::2, 2] = color_theme.piano_roll_background_color_even[2]
        pitches_alpha = np.ones((127), dtype=np.float32)
        pitches_edge_colors = np.zeros((127, 3), dtype=np.uint8)
        pitches_thickness = np.zeros((127), dtype=np.int32)
        canvas.draw_rectangles(
            pitches_rectangles,
            fill_colors=pitches_fill_colors,
            alpha=pitches_alpha,
            edge_colors=pitches_edge_colors,
            thickness=pitches_thickness,
        )

    # draw vertical lines every beat
    # get the beat positions
    subdivision_positions, beat_positions, bar_positions = score.get_subdivision_beat_and_bar_times()

    if grid_options.draw_subdivisions:
        # draw vertical lines every subdivision

        subdivision_rectangles = np.column_stack(
            (
                (subdivision_positions - pixel_mapping.time_left) * pixel_mapping.pixels_per_second,
                np.zeros_like(subdivision_positions),
                (subdivision_positions - pixel_mapping.time_left) * pixel_mapping.pixels_per_second + 1,
                np.full_like(subdivision_positions, height),
            )
        ).astype(np.int32)

        canvas.draw_rectangles(
            subdivision_rectangles,
            fill_colors=color_theme.subdivision_lines_color,
            alpha=1.0,
            edge_colors=None,
            thickness=0,
        )
    if grid_options.draw_beats:
        beat_rectangles = np.column_stack(
            (
                (beat_positions - pixel_mapping.time_left) * pixel_mapping.pixels_per_second,
                np.zeros_like(beat_positions),
                (beat_positions - pixel_mapping.time_left) * pixel_mapping.pixels_per_second + 1,
                np.full_like(beat_positions, height),
            )
        ).astype(np.int32)

        canvas.draw_rectangles(
            beat_rectangles,
            fill_colors=color_theme.beat_lines_color,
            alpha=1.0,
            edge_colors=None,
            thickness=0,
        )

    # draw vertical lines every bar
    bar_rectangles = np.column_stack(
        (
            (bar_positions - pixel_mapping.time_left) * pixel_mapping.pixels_per_second,
            np.zeros_like(bar_positions),
            (bar_positions - pixel_mapping.time_left) * pixel_mapping.pixels_per_second + 1,
            np.full_like(bar_positions, height),
        )
    ).astype(np.int32)

    canvas.draw_rectangles(
        bar_rectangles,
        fill_colors=color_theme.bar_lines_color,
        alpha=1.0,
        edge_colors=None,
        thickness=0,
    )


def piano_pitch_is_white(pitch: np.ndarray) -> np.ndarray:
    """Check if a pitch is a white key on a piano."""
    is_white_12 = np.ones((128,), dtype=bool)
    is_white_12[[1, 3, 6, 8, 10]] = False
    return is_white_12[pitch % 12]


def draw_piano_keys(canvas: CanvasProto, pitch_top: float, pitch_bottom: float) -> None:
    """Draw the piano keys.

    Args:
        canvas: The canvas to draw on.
        pitch_top: Top pitch value.
        pitch_bottom: Bottom pitch value.

    Returns:
        A drawing surface with piano keys.
    """
    height = canvas.height
    width = canvas.width
    pixels_per_pitch = height / (pitch_top - pitch_bottom)

    # Fill with white background
    bg_rect = np.array([[0, 0, width, height]], dtype=np.int32)
    canvas.draw_rectangles(bg_rect, fill_colors=(255, 255, 255), alpha=1.0, edge_colors=None, thickness=0)

    # draw the piano black and white keys
    # draw white keys
    pitches = np.arange(128)
    is_white_key = piano_pitch_is_white(pitches)
    white_keys_pitches = pitches[is_white_key]
    white_keys_edges = 0.5 * (white_keys_pitches[1:] + white_keys_pitches[:-1])

    white_keys_rectangles = np.column_stack(
        (
            np.zeros((len(white_keys_edges) - 1), dtype=np.int32),
            (pitch_top - white_keys_edges[1:]) * pixels_per_pitch,
            np.full((len(white_keys_edges) - 1), 30),
            1 + (pitch_top - white_keys_edges[:-1]) * pixels_per_pitch,
        )
    ).astype(np.int32)

    canvas.draw_rectangles(
        white_keys_rectangles, fill_colors=(255, 255, 255), alpha=1.0, edge_colors=(0, 0, 0), thickness=1
    )
    # draw black keys
    black_keys_pitches = pitches[~is_white_key]
    black_keys_rectangles = np.column_stack(
        (
            np.zeros((len(black_keys_pitches)), dtype=np.int32),
            (pitch_top - black_keys_pitches - 0.5) * pixels_per_pitch,
            np.full((len(black_keys_pitches)), 20),
            1 + (pitch_top - black_keys_pitches + 0.5) * pixels_per_pitch,
        )
    ).astype(np.int32)
    canvas.draw_rectangles(black_keys_rectangles, fill_colors=(0, 0, 0), alpha=1.0, edge_colors=None, thickness=0)


def draw_controls(
    canvas: CanvasProto,
    score: Score,
    track_number: int,
    control_number: int,
    time_left: float,
    time_right: float,
    color: Tuple[int, int, int],
    padding: int = 5,
) -> None:
    """Draw the controls for the loaded MIDI score."""
    height = canvas.height
    width = canvas.width
    controls = score.tracks[track_number].controls
    selected_controls = controls[controls.number == control_number]
    assert isinstance(selected_controls, ControlArray)

    # sort the controls by time
    selected_controls = selected_controls[np.argsort(selected_controls.time)]
    time = selected_controls.time
    value = selected_controls.value
    pixels_per_second = (width) / (time_right - time_left)
    x = (time - time_left) * pixels_per_second
    # add x corresponding to score.duration
    x_final = (score.duration - time_left) * pixels_per_second
    x = np.hstack((x, x_final))
    y = padding + (1 - value / 128) * (height - 2 * padding)
    draw_staircase(canvas, x=x, y=y, color=color)


def notes_to_rectangles(notes: NoteArray, box: PianorollBox, height: int, width: int) -> np.ndarray:
    """Convert notes to rectangles for drawing."""
    # Convert notes to rectangles for drawing
    pixel_mapping = get_pixel_mapping(width, height, box)
    rectangles = np.column_stack(
        (
            (notes.start - pixel_mapping.time_left) * pixel_mapping.pixels_per_second,
            (pixel_mapping.pitch_top - notes.pitch - 0.5) * pixel_mapping.pixels_per_pitch,
            1 + (notes.start + notes.duration - pixel_mapping.time_left) * pixel_mapping.pixels_per_second,
            ((pixel_mapping.pitch_top - notes.pitch) + 0.5) * pixel_mapping.pixels_per_pitch,
        )
    ).astype(np.int32)
    return rectangles


def draw_score_notes(
    canvas: CanvasProto,
    score: Score,
    box: PianorollBox,
    track_colors: TrackColors,
) -> None:
    for track_id, track in enumerate(score.tracks):
        draw_notes(
            canvas=canvas,
            notes=track.notes,
            box=box,
            fill_color=tuple(track_colors.fill_colors[track_id].tolist()),
            edge_color=tuple(track_colors.edge_colors[track_id].tolist()),
            thickness=track_colors.thickness.item(track_id),
            alpha=track_colors.alpha.item(track_id),
        )


def draw_notes(
    canvas: CanvasProto,
    notes: NoteArray,
    box: PianorollBox,
    fill_color: Tuple[int, int, int],
    edge_color: Tuple[int, int, int] | None = None,
    thickness: int = 0,
    alpha: float = 1.0,
) -> None:
    """Draw notes on the canvas."""
    if alpha == 0.0:
        return

    rectangles = notes_to_rectangles(notes=notes, box=box, height=canvas.height, width=canvas.width)
    canvas.draw_rectangles(
        rectangles,
        fill_colors=fill_color,
        alpha=alpha,  # Apply transparency to non-selected tracks
        edge_colors=edge_color,
        thickness=thickness,
    )


def draw_velocities(
    canvas: CanvasProto,
    score: Score,
    time_left: float,
    time_right: float,
    track_colors: TrackColors,
    velocity_max_width_pixels: float,
) -> None:
    """Draw the velocity of the loaded MIDI score."""
    for track_id, track in enumerate(score.tracks):
        track_color = tuple(track_colors.fill_colors[track_id].tolist())
        alpha = track_colors.alpha.item(track_id)
        edge_color = tuple(track_colors.edge_colors[track_id].tolist())
        thickness = track_colors.thickness.item(track_id)
        draw_notes_velocities(
            canvas=canvas,
            notes=track.notes,
            time_left=time_left,
            time_right=time_right,
            fill_color=track_color,
            edge_color=edge_color,
            thickness=thickness,
            alpha=alpha,
            velocity_max_width_pixels=velocity_max_width_pixels,
        )


def draw_notes_velocities(
    canvas: CanvasProto,
    notes: NoteArray,
    time_left: float,
    time_right: float,
    fill_color: Tuple[int, int, int],
    edge_color: Tuple[int, int, int] | None = None,
    thickness: int = 0,
    alpha: float = 1.0,
    velocity_max_width_pixels: float = 10.0,
) -> None:
    """Draw the velocities of a track."""
    if alpha == 0.0:
        return

    height, width = canvas.height, canvas.width
    velocities = notes.velocity
    start_times = notes.start
    pixels_per_second = width / (time_right - time_left)
    end_times = notes.start + np.minimum(notes.duration, velocity_max_width_pixels / pixels_per_second)

    rectangles = np.column_stack(
        (
            (start_times - time_left) * pixels_per_second,
            (1 - velocities / 128) * height,
            1 + (end_times - time_left) * pixels_per_second,
            np.full_like(start_times, canvas.height),
        )
    ).astype(np.int32)

    canvas.draw_rectangles(
        rectangles,
        fill_colors=fill_color,
        alpha=alpha,
        edge_colors=edge_color,
        thickness=thickness,
    )


def draw_staircase(
    canvas: CanvasProto,
    x: np.ndarray,
    y: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 1,
    alpha: float = 1.0,
) -> None:
    # create the polyline
    # [[x[0],y[0]], [x[1], y[0]], [x[1], y[1]], ...., [x[n-1], y[n-1]], [x[n], y[n-1]]]
    staircase_x = np.repeat(x, 2)[1:-1]
    staircase_y = np.repeat(y, 2)

    canvas.draw_polyline(
        x=staircase_x,
        y=staircase_y,
        color=color,
        thickness=thickness,
        alpha=alpha,
    )


def draw_tempo(
    canvas: CanvasProto,
    score: Score,
    color: Tuple[int, int, int],
    time_left: float,
    time_right: float,
    qnpm_max: float,
    qnpm_min: float,
) -> None:
    time = score.tempo.time
    height = canvas.height
    width = canvas.width
    quarter_notes_per_minute = score.tempo.quarter_notes_per_minute
    pixels_per_second = (width - 1) / (time_right - time_left)
    x = (time - time_left) * pixels_per_second  # add x corresponding to score.duration
    x_final = (score.duration - time_left) * pixels_per_second
    x = np.hstack((x, x_final))
    y = height - 1 - (quarter_notes_per_minute - qnpm_min) / (qnpm_max - qnpm_min) * (height - 1)
    draw_staircase(canvas, x=x, y=y, color=color)


def draw_pitch_bends(
    canvas: CanvasProto,
    score: Score,
    track_number: int,
    time_left: float,
    time_right: float,
    color: Tuple[int, int, int],
    padding: int,
) -> None:
    track = score.tracks[track_number]

    pixels_per_second = (canvas.width - 1) / (time_right - time_left)
    # sort the pitch_bends by tick
    pitch_bends = track.pitch_bends[np.argsort(track.pitch_bends.tick)]
    assert np.all(np.diff(pitch_bends.tick) > 0), "Pitch bends must be sorted by tick"
    time = pitch_bends.time
    value = pitch_bends.value

    x = (time - time_left) * pixels_per_second  # add x corresponding to score.duration
    x_final = (score.duration - time_left) * pixels_per_second
    x = np.hstack((x, x_final))

    height = canvas.height - padding
    # Normalize value to the range of 0 to canvas_height
    y = ((8192 - value) / 16384) * height + 0.5 * padding

    draw_staircase(canvas, x=x, y=y, color=color)
