"""Drawing backends for MidiDraw.

This module provides abstract and concrete drawing backends for the MidiDraw class.
Each backend implements a common interface for drawing operations like rectangles and lines.
"""

from typing import Optional, Tuple, Union

import numpy as np

from numba_midi.numba_draw import draw_polyline, draw_rectangles


class NumPyCanvas:
    """Drawing backend that uses NumPy arrays and numba-accelerated functions."""

    def __init__(self, height: int, width: int) -> None:
        """Initialize the NumPy drawing backend."""
        self.height = height
        self.width = width
        self._image = np.zeros((height, width, 3), dtype=np.uint8)

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
        """Draw rectangles on the NumPy surface using numba-accelerated function."""
        draw_rectangles(
            image=self._image,
            rectangles=rectangles,
            fill_colors=fill_colors,
            alpha=alpha,
            edge_colors=edge_colors,
            thickness=thickness,
            fill_alpha=fill_alpha,
            edge_alpha=edge_alpha,
            prefilter=True,
        )

    def draw_polyline(
        self,
        x: np.ndarray,
        y: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int = 1,
        alpha: float = 1.0,
    ) -> None:
        """Draw a polyline on the NumPy surface using numba-accelerated function."""
        # Current implementation doesn't support thickness and alpha
        draw_polyline(
            image=self._image,
            x=x,
            y=y,
            color=color,
        )

    def clear(self, color: Tuple[int, int, int]) -> None:
        """Clear the canvas with a specific color."""
        self._image[:, :, 0] = color[0]
        self._image[:, :, 1] = color[1]
        self._image[:, :, 2] = color[2]
