"""Draw rectangles and polylines on an image using Numba for performance."""

from typing import Tuple

import numpy as np

# from numba_midi.numba.draw import draw_polyline_jit, draw_rectangles_jit
import numba_midi.cython.draw as draw_acc


class NumPyCanvas:
    """Drawing backend that uses NumPy arrays and numba-accelerated functions."""

    def __init__(self, height: int, width: int) -> None:
        """Initialize the NumPy drawing backend."""
        self.height = height
        self.width = width
        self._image = np.zeros((height, width, 3), dtype=np.uint8)

    def draw_rectangles(self, rectangles: "Rectangles") -> None:
        """Draw rectangles on the NumPy surface using numba-accelerated function."""
        rectangles.draw(self._image)

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


class Rectangles:
    """A class to represent a collection of rectangles with colors and alpha values."""

    def __init__(
        self,
        corners: np.ndarray,
        fill_colors: np.ndarray | tuple[int, int, int],
        fill_alpha: np.ndarray | float = 1.0,
        edge_colors: np.ndarray | tuple[int, int, int] | None = None,
        thickness: np.ndarray | int = 0,
        edge_alpha: np.ndarray | float | None = None,
    ) -> None:
        num_rectangles = corners.shape[0]

        if edge_alpha is None:
            edge_alpha = fill_alpha

        assert corners.ndim == 2 and corners.shape[1] == 4, "corners must be a 2D array with 4 columns (x1, y1, x2, y2)"
        self.corners = corners

        # Convert single values to arrays

        if isinstance(fill_alpha, float):
            self.fill_alpha = np.full(num_rectangles, fill_alpha, dtype=np.float32)
        else:
            assert fill_alpha.shape[0] == num_rectangles, "fill_alpha must have the same number of rows as rectangles"
            self.fill_alpha = fill_alpha

        if isinstance(edge_alpha, float):
            self.edge_alpha = np.full(num_rectangles, edge_alpha, dtype=np.float32)
        else:
            assert edge_alpha.shape[0] == num_rectangles, "edge_alpha must have the same number of rows as rectangles"
            self.edge_alpha = edge_alpha

        if isinstance(thickness, int):
            self.thickness = np.full(num_rectangles, thickness, dtype=np.int32)
        else:
            assert thickness.shape[0] == num_rectangles, "thickness must have the same number of rows as rectangles"
            self.thickness = thickness

        # Convert single color values to arrays
        if isinstance(fill_colors, tuple):
            self.fill_colors = np.tile(np.array(fill_colors, dtype=np.uint8), (num_rectangles, 1))
        else:
            assert fill_colors.ndim == 2 and fill_colors.shape[1] == 3, (
                "fill_colors must be a 2D array with 3 columns (R, G, B)"
            )
            self.fill_colors = fill_colors
        assert self.fill_colors.shape[0] == num_rectangles, (
            "fill_colors must have the same number of rows as rectangles"
        )
        assert self.fill_colors.dtype == np.uint8, "fill_colors must be of type np.uint8"
        # Handle edge colors
        if edge_colors is None:
            self.edge_colors = np.zeros_like(self.fill_colors, dtype=np.uint8)
        elif isinstance(edge_colors, tuple):
            self.edge_colors = np.tile(np.array(edge_colors, dtype=np.uint8), (num_rectangles, 1))
        else:
            assert edge_colors.ndim == 2 and edge_colors.shape[1] == 3, (
                "edge_colors must be a 2D array with 3 columns (R, G, B)"
            )
            self.edge_colors = edge_colors
        assert self.edge_colors.dtype == np.uint8, "edge_colors must be of type np.uint8"
        assert self.fill_alpha.dtype == np.float32, "fill_alpha must be of type np.float32"
        assert self.edge_alpha.dtype == np.float32, "edge_alpha must be of type np.float32"
        assert self.thickness.dtype == np.int32, "thickness must be of type np.int32"

    def filter_box(
        self,
        height: int,
        width: int,
    ) -> "Rectangles":
        """Filter rectangles that are completely outside the image bounds."""
        keep_mask = (
            (self.corners[:, 0] < width)
            & (self.corners[:, 2] > 0)
            & (self.corners[:, 1] < height)
            & (self.corners[:, 3] > 0)
        )
        return Rectangles(
            corners=self.corners[keep_mask],
            fill_colors=self.fill_colors[keep_mask],
            fill_alpha=self.fill_alpha[keep_mask],
            edge_colors=self.edge_colors[keep_mask],
            thickness=self.thickness[keep_mask],
            edge_alpha=self.edge_alpha[keep_mask],
        )

    def draw(
        self,
        image: np.ndarray,
        prefilter: bool = True,
    ) -> np.ndarray:
        """Draw rectangles on an image with separate fill and edge alpha values."""
        if prefilter:
            rectangles = self.filter_box(image.shape[0], image.shape[1])
            rectangles.draw(image, prefilter=False)
            return image

        assert image.dtype == np.uint8

        draw_acc.draw_rectangles(
            image=image,
            rectangles=self.corners,
            fill_colors=self.fill_colors,
            fill_alpha=self.fill_alpha,
            edge_colors=self.edge_colors,
            edge_alpha=self.edge_alpha,
            thickness=self.thickness,
        )
        return image


def draw_polyline(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    color: np.ndarray | tuple[int, int, int],
) -> np.ndarray:
    """Draw a polyline on an image.

    x and y are 1D arrays with the same length.
    """
    if isinstance(color, tuple):
        color = np.array(color, dtype=np.uint8)

    draw_acc.draw_polyline(image, x, y, color)
    return image
