"""A tkinter GUI for editing MIDI files.

This module provides a graphical user interface (GUI) for editing MIDI files using the tkinter library.
"""

import copy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import tkinter as tk
from tkinter import colorchooser, filedialog
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageTk

from numba_midi import load_score
from numba_midi.editor.numba_draw import draw_rectangles, draw_starcaise
from numba_midi.instruments import all_instruments
from numba_midi.score import Score

# 16 track colors with more distinct tones
default_track_colors = [
    (255, 99, 71),  # Tomato
    (60, 179, 113),  # Medium Sea Green
    (30, 144, 255),  # Dodger Blue
    (255, 215, 0),  # Gold
    (138, 43, 226),  # Blue Violet
    (0, 206, 209),  # Dark Turquoise
    (169, 169, 169),  # Dark Gray
    (220, 20, 60),  # Crimson
    (50, 205, 50),  # Lime Green
    (70, 130, 180),  # Steel Blue
    (255, 140, 0),  # Dark Orange
    (186, 85, 211),  # Medium Orchid
    (0, 255, 127),  # Spring Green
    (64, 224, 208),  # Turquoise
    (255, 165, 0),  # Orange
    (75, 0, 130),  # Indigo
]


def piano_pitch_is_white(pitch: np.ndarray) -> bool:
    """Check if a pitch is a white key on a piano."""
    is_white_12 = np.ones((128,), dtype=bool)
    is_white_12[[1, 3, 6, 8, 10]] = False

    return is_white_12[pitch % 12]


@dataclass
class EditorState:
    """A class representing the state of the MIDI editor."""

    selected_notes: list = field(default_factory=list)
    pixels_per_second: float = 10
    pixels_per_pitch: float = 5
    selected_track: int = 0
    time_left: float = 0.0
    pitch_top: float = 127.0
    button_pressed: bool = False
    button_pressed_xy: Optional[tuple[float, float]] = None
    button_pressed_time_left: Optional[float] = None
    button_pressed_pitch_top: Optional[float] = None
    keys_pressed: set[str] = field(default_factory=set)
    selected_tracks: set[int] = field(default_factory=set)
    track_colors: list[Tuple[int, int, int]] = field(default_factory=lambda: copy.copy(default_track_colors))
    notes_edge_color: Tuple[int, int, int] = (180, 180, 180)  # light grey


class MidiEditor(tk.Tk):
    """A class representing a MIDI editor GUI.

    This class provides a graphical user interface for editing MIDI files using the tkinter library.
    It allows users to load, save, and edit MIDI files.

    Attributes:
        score (Score): The loaded MIDI score.
        midi_file (Path): The path to the loaded MIDI file.

    """

    def __init__(self, midi_file: Optional[Path] = None) -> None:
        """Initialize the MidiEditor class.

        This method initializes the tkinter GUI and sets up the necessary widgets and layout.
        """
        super().__init__()
        self.title("MIDI Editor")
        self.geometry("800x600")

        self.score: Optional[Score] = None
        self.midi_file: Optional[Path] = None

        self.editor_state = EditorState()

        # add "File" menu
        self.menu = tk.Menu(self)
        self.config(menu=self.menu)
        self.file_menu = tk.Menu(self.menu, tearoff=0)
        # add Open in the file menu
        self.file_menu.add_command(label="Open", command=self.open_file)

        # add File menu to the menu bar
        self.menu.add_cascade(label="File", menu=self.file_menu)

        # slit the main window in two parts left is panaio roll and right is the track list
        self.pianoroll_frame = tk.Frame(self)
        self.pianoroll_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.pianoroll_canvas = tk.Canvas(self.pianoroll_frame, bg="white")
        self.pianoroll_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.velocity_canvas = tk.Canvas(self.pianoroll_frame, bg="white", height=50)
        self.velocity_canvas.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

        self.controls_canvas = tk.Canvas(self.pianoroll_frame, bg="white", height=50)
        self.controls_canvas.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

        # call resfresh if there is the windows is resized
        self.pianoroll_canvas.bind("<Configure>", lambda event: self.refresh())

        # add callback to move the view when the mouse is dragged
        self.pianoroll_canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.pianoroll_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.pianoroll_canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.pianoroll_canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        # add right panel to select the tracks to display
        self.track_frame = tk.Frame(self)
        self.track_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # add a canvas and scrollbar for the track list
        self.track_canvas = tk.Canvas(self.track_frame, width=200)  # Set a fixed width for the track frame
        self.track_scrollbar = tk.Scrollbar(self.track_frame, orient=tk.VERTICAL, command=self.track_canvas.yview)
        self.track_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.track_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        # configure the canvas to work with the scrollbar
        self.track_canvas.configure(yscrollcommand=self.track_scrollbar.set)
        self.track_canvas.bind(
            "<Configure>", lambda e: self.track_canvas.configure(scrollregion=self.track_canvas.bbox("all"))
        )

        # create a frame inside the canvas for the track list
        self.tracks_list = tk.Frame(self.track_canvas)
        self.track_canvas.create_window((0, 0), window=self.tracks_list, anchor="nw")

        # key press events
        self.bind("<KeyPress>", self.on_key_press)
        self.bind("<KeyRelease>", self.on_key_release)

        self.velocity_canvas_image: Optional[ImageTk.PhotoImage] = None
        self.controls_canvas_image: Optional[ImageTk.PhotoImage] = None
        self.pianoroll_canvas_image: Optional[ImageTk.PhotoImage] = None

    def on_key_press(self, event: tk.Event) -> None:
        """Handle key press events.

        This method is called when a key is pressed.

        Args:
            event (tk.Event): The key event.

        """
        if event.keysym == "Left":
            self.move_left()
        elif event.keysym == "Right":
            self.move_right()
        elif event.keysym == "Up":
            self.move_up()
        elif event.keysym == "Down":
            self.move_down()
        elif event.keysym == "plus":
            self.zoom_in()
        elif event.keysym == "minus":
            self.zoom_out()

        self.editor_state.keys_pressed.add(event.keysym)

    def on_key_release(self, event: tk.Event) -> None:
        """Handle key release events.

        This method is called when a key is released.

        Args:
            event (tk.Event): The key event.

        """
        if event.keysym in self.editor_state.keys_pressed:
            self.editor_state.keys_pressed.remove(event.keysym)

    def on_button_press(self, event: tk.Event) -> None:
        """Handle mouse button press events.

        This method is called when the left mouse button is pressed.

        Args:
            event (tk.Event): The mouse event.

        """
        self.editor_state.button_pressed = True
        self.editor_state.button_pressed_xy = (event.x, event.y)
        self.editor_state.button_pressed_time_left = self.editor_state.time_left
        self.editor_state.button_pressed_pitch_top = self.editor_state.pitch_top

    def on_mouse_drag(self, event: tk.Event) -> None:
        """Handle mouse drag events.

        This method is called when the left mouse button is dragged.

        Args:
            event (tk.Event): The mouse event.

        """
        pass
        if self.editor_state.button_pressed:
            assert self.editor_state.button_pressed_xy is not None
            assert self.editor_state.button_pressed_time_left is not None
            assert self.editor_state.button_pressed_pitch_top is not None

            dx = event.x - self.editor_state.button_pressed_xy[0]
            dy = event.y - self.editor_state.button_pressed_xy[1]

            self.editor_state.time_left = (
                self.editor_state.button_pressed_time_left - dx / self.editor_state.pixels_per_second
            )
            self.editor_state.pitch_top = (
                self.editor_state.button_pressed_pitch_top + dy / self.editor_state.pixels_per_pitch
            )
            self.refresh()

    def on_button_release(self, event: tk.Event) -> None:
        """Handle mouse button release events.

        This method is called when the left mouse button is released.

        Args:
            event (tk.Event): The mouse event.

        """
        self.editor_state.button_pressed = False
        self.editor_state.button_pressed_xy = None

    def on_mouse_wheel(self, event: tk.Event) -> None:
        """Handle mouse wheel events.

        This method is called when the mouse wheel is scrolled.

        Args:
            event (tk.Event): The mouse event.

        """
        if event.delta > 0:
            # if alt is pressed, zoom in the pitch axis
            if "Control_L" in self.editor_state.keys_pressed:
                self.zoom_in_pitch()
            else:
                # zoom in the time axis
                self.zoom_in(x=event.x)

        elif "Control_L" in self.editor_state.keys_pressed:
            self.zoom_out_pitch()
        else:
            self.zoom_out(x=event.x)

    def zoom_in(self, x: Optional[float] = None) -> None:
        """Zoom in the time axis.

        This method increases the pixels per second to zoom in the time axis.

        """
        if self.score is None:
            return
        if x is None:
            x = 0.5 * self.pianoroll_canvas.winfo_width()
        time_center = self.editor_state.time_left + x / self.editor_state.pixels_per_second
        time_center = min(time_center, self.score.duration)
        self.editor_state.pixels_per_second *= 1.1
        self.editor_state.time_left = time_center - x / self.editor_state.pixels_per_second
        self.editor_state.time_left = max(0, self.editor_state.time_left)
        self.refresh()

    def zoom_out(self, x: Optional[float] = None) -> None:
        """Zoom out the time axis.

        This method decreases the pixels per second to zoom out the time axis.

        """
        if x is None:
            x = 0.5 * self.pianoroll_canvas.winfo_width()
        time_center = self.editor_state.time_left + x / self.editor_state.pixels_per_second
        self.editor_state.pixels_per_second /= 1.1
        self.editor_state.time_left = time_center - x / self.editor_state.pixels_per_second
        self.editor_state.time_left = max(0, self.editor_state.time_left)
        self.refresh()

    def zoom_in_pitch(self) -> None:
        """Zoom in the pitch axis.

        This method increases the pixels per pitch to zoom in the pitch axis.

        """
        pitch_center = self.editor_state.pitch_top - self.pianoroll_canvas.winfo_height() / (
            2 * self.editor_state.pixels_per_pitch
        )
        self.editor_state.pixels_per_pitch *= 1.1
        self.editor_state.pitch_top = pitch_center + self.pianoroll_canvas.winfo_height() / (
            2 * self.editor_state.pixels_per_pitch
        )
        # self.state.pitch_top = min(127, self.state.pitch_top)
        self.refresh()

    def zoom_out_pitch(self) -> None:
        """Zoom out the pitch axis.

        This method decreases the pixels per pitch to zoom out the pitch axis.

        """
        pitch_center = self.editor_state.pitch_top - self.pianoroll_canvas.winfo_height() / (
            2 * self.editor_state.pixels_per_pitch
        )
        self.editor_state.pixels_per_pitch /= 1.1
        self.editor_state.pitch_top = pitch_center + self.pianoroll_canvas.winfo_height() / (
            2 * self.editor_state.pixels_per_pitch
        )
        # self.state.pitch_top = min(127, self.state.pitch_top)
        self.refresh()

    def move_left(self) -> None:
        """Move the view to the left.

        This method decreases the time left to move the view to the left.

        """
        self.editor_state.time_left -= 10 / self.editor_state.pixels_per_second
        self.refresh()

    def move_right(self) -> None:
        """Move the view to the right.

        This method increases the time left to move the view to the right.

        """
        self.editor_state.time_left += 10 / self.editor_state.pixels_per_second
        self.refresh()

    def move_up(self) -> None:
        """Move the view up.

        This method increases the pitch top to move the view up.

        """
        self.editor_state.pitch_top += 1
        self.refresh()

    def move_down(self) -> None:
        """Move the view down.

        This method decreases the pitch top to move the view down.

        """
        self.editor_state.pitch_top -= 1
        self.refresh()

    def refresh(self) -> None:
        """Refresh the editor.

        This method refreshes the editor by redrawing the pianoroll and updating the canvas.

        """
        self.draw_pianoroll()
        self.draw_velocity()
        self.draw_controls()

    def open_file(self) -> None:
        """Open a MIDI file.

        This method opens a file dialog to select a MIDI file and loads it into the editor.

        """
        file_path = filedialog.askopenfilename(filetypes=[("MIDI files", "*.mid")])
        if file_path:
            self.midi_file = Path(file_path)
            self.score = load_score(self.midi_file, notes_mode="no_overlap")
        assert self.score is not None, "Failed to load MIDI file."
        self.editor_state.selected_tracks = set()
        self.track_color_boxes: dict[int, tk.Frame] = {}
        self.track_toggle_buttons: dict[int, tk.Button] = {}
        for i, track in enumerate(self.score.tracks):
            if not track.name:
                track_name = f"Track {i}"
            else:
                track_name = track.name

            instrument_name = all_instruments[track.program]

            name = f"{track_name} ({instrument_name})"
            # create a toggle button for each track
            # use text color to show the track color
            track_color = self.editor_state.track_colors[i % len(self.editor_state.track_colors)]
            track_color_box = tk.Frame(
                self.tracks_list,
                bg=f"#{track_color[0]:02x}{track_color[1]:02x}{track_color[2]:02x}",
                width=20,
                height=20,
            )
            self.track_color_boxes[i] = track_color_box
            # add color picker to the track color box

            track_color_box.bind("<Button-1>", partial(self.change_track_color, i))

            track_toggle = tk.Button(
                self.tracks_list,
                text=name,
                command=partial(self.toggle_track, i),
                bg="white",
                fg="black",
                font=("TkDefaultFont", 8),
                height=1,
            )
            self.track_toggle_buttons[i] = track_toggle
            # add the toggle button and color box to the track list
            track_color_box.grid(row=i, column=0, padx=5, pady=5, sticky=tk.W)
            track_toggle.grid(row=i, column=1, padx=5, pady=5, sticky=tk.W + tk.E)
            self.tracks_list.grid_columnconfigure(1, weight=1)
            self.editor_state.selected_tracks.add(i)  # add the track to the selected tracks

        self.refresh()

    def change_track_color(self, track_id: int) -> None:
        """Change the color of a track.

        This method opens a color picker dialog to select a new color for the track.

        Args:
            track_id (int): The ID of the track to change the color.

        """
        # open a color picker dialog
        color = colorchooser.askcolor()[1]
        if color:
            # set the color of the track
            track_color = (int(color[1:2], 16), int(color[3:5], 16), int(color[5:7], 16))
            # update the track color in the GUI
            self.editor_state.track_colors[track_id] = track_color
            track_color_str = f"#{track_color[0]:02x}{track_color[1]:02x}{track_color[2]:02x}"
            self.track_color_boxes[track_id].config(bg=track_color_str)

        self.refresh()

    def toggle_track(self, track_id: int) -> None:
        """Toggle the display of a track.

        This method toggles the display of a track in the pianoroll.

        Args:
            track_id (int): The ID of the track to toggle.

        """
        if self.score is None:
            return

        # toggle the display of the track
        if track_id in self.editor_state.selected_tracks:
            self.editor_state.selected_tracks.remove(track_id)
            # make the text in the button light grey
            self.track_toggle_buttons[track_id].config(fg="grey")

        else:
            self.editor_state.selected_tracks.add(track_id)
            # make the text in the button black
            self.track_toggle_buttons[track_id].config(fg="black")

        # redraw the pianoroll
        self.refresh()

    def draw_velocity(self) -> None:
        """Draw the velocity of the loaded MIDI score.

        Args:
            score (Score): The loaded MIDI score.

        """
        if self.score is None:
            return
        height = self.velocity_canvas.winfo_height()
        image = np.full((height, self.velocity_canvas.winfo_width(), 3), 255, dtype=np.uint8)

        for track_id in self.editor_state.selected_tracks:
            track = self.score.tracks[track_id]
            track_color = self.editor_state.track_colors[track_id % len(self.editor_state.track_colors)]

            velocities = track.notes["velocity_on"]
            start_times = track.notes["start"]
            velocity_max_width_pixels = 10
            end_times = track.notes["start"] + np.minimum(
                track.notes["duration"], velocity_max_width_pixels / self.editor_state.pixels_per_second
            )
            rectangles = np.column_stack(
                (
                    (start_times - self.editor_state.time_left) * self.editor_state.pixels_per_second,
                    (1 - velocities / 128) * height,
                    1 + (end_times - self.editor_state.time_left) * self.editor_state.pixels_per_second,
                    np.full_like(start_times, height),
                )
            ).astype(np.int32)
            # draw the rectangles
            draw_rectangles(
                image,
                rectangles,
                fill_colors=track_color,
                alpha=1.0,
                edge_colors=self.editor_state.notes_edge_color,
                thickness=1,
            )
        # put the image on the canvas
        self.velocity_canvas.delete("all")
        img = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.velocity_canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.velocity_canvas_image = img  # keep a reference to the image

    def draw_controls(self) -> None:
        """Draw the controls for the loaded MIDI score.

        Args:
            score (Score): The loaded MIDI score.

        """
        if self.score is None:
            return
        image = np.full(
            (self.controls_canvas.winfo_height(), self.controls_canvas.winfo_width(), 3), 255, dtype=np.uint8
        )

        for track_id in self.editor_state.selected_tracks:
            track = self.score.tracks[track_id]
            track_color = self.editor_state.track_colors[track_id % len(self.editor_state.track_colors)]

            # plot pitch bends
            if len(track.pitch_bends) == 0:
                continue
            time = track.pitch_bends["time"]
            value = track.pitch_bends["value"]

            x = (time - self.editor_state.time_left) * self.editor_state.pixels_per_second
            # add x correspondng to scrore.duration
            x_final = (self.score.duration - self.editor_state.time_left) * self.editor_state.pixels_per_second
            x = np.hstack((x, x_final))
            y = 5 + (1 - value / 128) * (self.controls_canvas.winfo_height() - 10)
            draw_starcaise(image, x=x, y=y, color=track_color)
        # put the image on the canvas

        img = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.controls_canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.controls_canvas_image = img  # keep a reference to the image

    def draw_pianoroll(self) -> None:
        """Draw the pianoroll of the loaded MIDI score.

        Args:
            score (Score): The loaded MIDI score.

        """
        # Clear the canvas
        if self.score is None:
            return

        # get the size of the canvas
        width = self.pianoroll_canvas.winfo_width()
        height = self.pianoroll_canvas.winfo_height()

        # create an empty image as numpy array
        image = np.full((height, width, 3), 255, dtype=np.uint8)

        pixels_per_second = self.editor_state.pixels_per_second
        pixels_per_pitch = self.editor_state.pixels_per_pitch
        time_left = self.editor_state.time_left

        # draw light grey rectangle every other pitch
        pitches_rectangles = np.column_stack(
            (
                np.zeros((127), dtype=np.int32),
                (self.editor_state.pitch_top - np.arange(127) - 0.5) * pixels_per_pitch,
                np.full((127), width),
                (self.editor_state.pitch_top - np.arange(127) + 0.5) * pixels_per_pitch,
            )
        ).astype(np.int32)

        pitches_fill_colors = np.full((127, 3), 255, dtype=np.uint8)
        pitches_fill_colors[::2, 0] = 230
        pitches_fill_colors[::2, 1] = 230
        pitches_fill_colors[::2, 2] = 230
        pitches_alpha = np.ones((127), dtype=np.float32)
        pitches_edge_colors = np.zeros((127, 3), dtype=np.uint8)
        pitches_thickness = np.zeros((127), dtype=np.int32)
        draw_rectangles(
            image,
            pitches_rectangles,
            fill_colors=pitches_fill_colors,
            alpha=pitches_alpha,
            edge_colors=pitches_edge_colors,
            thickness=pitches_thickness,
        )

        # draw vertical lines every beat
        # get the beat positions
        beat_positions = self.score.get_beat_positions()

        beat_rectangles = np.column_stack(
            (
                (beat_positions - time_left) * pixels_per_second,
                np.zeros_like(beat_positions),
                (beat_positions - time_left) * pixels_per_second + 1,
                np.full_like(beat_positions, height),
            )
        ).astype(np.int32)

        draw_rectangles(image, beat_rectangles, fill_colors=(220, 220, 220), alpha=1.0, edge_colors=None, thickness=0)
        # draw vertical lines every bar
        # get the bar positions
        bar_positions = self.score.get_bar_positions()
        bar_rectangles = np.column_stack(
            (
                (bar_positions - time_left) * pixels_per_second,
                np.zeros_like(bar_positions),
                (bar_positions - time_left) * pixels_per_second + 1,
                np.full_like(bar_positions, height),
            )
        ).astype(np.int32)

        draw_rectangles(image, bar_rectangles, fill_colors=(210, 210, 210), alpha=1.0, edge_colors=None, thickness=0)

        # create the rectangles to draw for the notes
        # using numpy vectorized operations
        for track_id, track in enumerate(self.score.tracks):
            if track_id not in self.editor_state.selected_tracks:
                continue
            track_color = self.editor_state.track_colors[track_id % len(self.editor_state.track_colors)]

            rectangles = np.column_stack(
                (
                    (track.notes["start"] - time_left) * pixels_per_second,
                    (self.editor_state.pitch_top - track.notes["pitch"] - 0.5) * pixels_per_pitch,
                    1 + (track.notes["start"] + track.notes["duration"] - time_left) * pixels_per_second,
                    ((self.editor_state.pitch_top - track.notes["pitch"]) + 0.5) * pixels_per_pitch,
                )
            ).astype(np.int32)
            draw_rectangles(
                image,
                rectangles,
                fill_colors=track_color,
                alpha=1.0,
                edge_colors=self.editor_state.notes_edge_color,
                thickness=1,
            )

        # draw the piano black and white keys
        # draw white keys
        pitches = np.arange(128)
        is_white_key = piano_pitch_is_white(pitches)
        white_keys_pitches = pitches[is_white_key]
        white_keys_edges = 0.5 * (white_keys_pitches[1:] + white_keys_pitches[:-1])

        white_keys_rectangles = np.column_stack(
            (
                np.zeros((len(white_keys_edges) - 1), dtype=np.int32),
                (self.editor_state.pitch_top - white_keys_edges[1:]) * pixels_per_pitch,
                np.full((len(white_keys_edges) - 1), 30),
                1 + (self.editor_state.pitch_top - white_keys_edges[:-1]) * pixels_per_pitch,
            )
        ).astype(np.int32)

        draw_rectangles(
            image, white_keys_rectangles, fill_colors=(255, 255, 255), alpha=1.0, edge_colors=(0, 0, 0), thickness=1
        )
        # draw black keys
        black_keys_pitches = pitches[~is_white_key]
        black_keys_rectangles = np.column_stack(
            (
                np.zeros((len(black_keys_pitches)), dtype=np.int32),
                (self.editor_state.pitch_top - black_keys_pitches - 0.5) * pixels_per_pitch,
                np.full((len(black_keys_pitches)), 20),
                1 + (self.editor_state.pitch_top - black_keys_pitches + 0.5) * pixels_per_pitch,
            )
        ).astype(np.int32)
        draw_rectangles(image, black_keys_rectangles, fill_colors=(0, 0, 0), alpha=1.0, edge_colors=None, thickness=0)

        # put the image on the canvas
        self.pianoroll_canvas.delete("all")
        img = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.pianoroll_canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.pianoroll_canvas_image = img  # keep a reference to the image

        # refresh the canvas
        # self.canvas.update_idletasks()


if __name__ == "__main__":
    editor = MidiEditor()
    editor.mainloop()
