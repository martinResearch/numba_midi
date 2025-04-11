"""A tkinter GUI for editing MIDI files.

This module provides a graphical user interface (GUI) for editing MIDI files using the tkinter library.
"""

from numba_midi import load_score
from numba_midi.score import Score

from dataclasses import dataclass

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from typing import Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
from numba_midi.editor.numba_draw import draw_rectangles
from PIL import Image, ImageTk
import time


@dataclass
class EditorState:
    selected_notes: list = field(default_factory=list)
    selected_track: int = 0


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

        # add "File" menu
        self.menu = tk.Menu(self)
        self.config(menu=self.menu)
        self.file_menu = tk.Menu(self.menu, tearoff=0)
        # add Open in the file menu
        self.file_menu.add_command(label="Open", command=self.open_file)

        # add File menu to the menu bar
        self.menu.add_cascade(label="File", menu=self.file_menu)

        # Create widgets
        # self.create_widgets()
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # call resfresh if there is the windows is resized
        self.canvas.bind("<Configure>", lambda event: self.refresh())

    def refresh(self) -> None:
        """Refresh the editor.

        This method refreshes the editor by redrawing the pianoroll and updating the canvas.

        """
        self.draw_pianoroll()

    def open_file(self) -> None:
        """Open a MIDI file.

        This method opens a file dialog to select a MIDI file and loads it into the editor.

        """
        file_path = filedialog.askopenfilename(filetypes=[("MIDI files", "*.mid")])
        if file_path:
            self.midi_file = Path(file_path)
            self.score = load_score(self.midi_file, notes_mode="no_overlap")
            self.draw_pianoroll()

    def draw_pianoroll(self) -> None:
        """Draw the pianoroll of the loaded MIDI score.

        Args:
            score (Score): The loaded MIDI score.

        """
        # Clear the canvas
        if self.score is None:
            return

        # get the size of the canvas
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        # create an empty image as numpy array
        image = np.full((height, width, 3), 255, dtype=np.uint8)

        pixels_per_second = 10
        pixels_per_pitch = 5

        # create the rectangles to draw for the notes
        # using numpy vectorized operations
        for track in self.score.tracks:
            num_notes = len(track.notes)
            rectangles = np.zeros((len(track.notes), 4), dtype=np.int32)
            rectangles[:, 0] = track.notes["start"] * pixels_per_second
            rectangles[:, 1] = track.notes["pitch"] * pixels_per_pitch
            rectangles[:, 2] = (track.notes["start"] + track.notes["duration"]) * pixels_per_second
            rectangles[:, 3] = (track.notes["pitch"] + 1) * pixels_per_pitch

            fill_colors = np.zeros((num_notes, 3), dtype=np.uint8)
            fill_colors[:, 0] = 255  # Red
            fill_colors[:, 1] = 0  # Green
            fill_colors[:, 2] = 0  # Blue

            alpha = np.ones((num_notes), dtype=np.float32)

            edge_colors = np.zeros((num_notes, 3), dtype=np.uint8)
            thickness = np.ones((num_notes), dtype=np.int32) * 1

            draw_rectangles(
                image, rectangles, fill_colors=fill_colors, alpha=alpha, edge_colors=edge_colors, thickness=thickness
            )

        # put the image on the canvas
        self.canvas.delete("all")
        img = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img  # keep a reference to the image

        # refresh the canvas
        # self.canvas.update_idletasks()


if __name__ == "__main__":
    editor = MidiEditor()
    editor.mainloop()
