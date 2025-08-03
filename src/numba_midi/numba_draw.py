"""Draw rectangles and polylines on an image using Numba for performance."""

from typing import Tuple

from numba.core.decorators import njit
import numpy as np


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


@njit(cache=True, boundscheck=False, nogil=True, fastmath=True)
def draw_rectangle_no_edge_jit(
    image: np.ndarray, color: np.ndarray, x1: int, x2: int, y1: int, y2: int, alpha: float
) -> None:
    if alpha == 1.0:
        image[y1:y2, x1:x2] = alpha * color
    else:
        image[y1:y2, x1:x2] = alpha * color + (1 - alpha) * image[y1:y2, x1:x2]


@njit(cache=True, boundscheck=False, nogil=True, fastmath=True)
def draw_rectangles_jit(
    image: np.ndarray,
    rectangles: np.ndarray,
    fill_colors: np.ndarray,
    fill_alpha: np.ndarray,
    thickness: np.ndarray,
    edge_colors: np.ndarray,
    edge_alpha: np.ndarray,
) -> None:
    """Draw rectangles on an image with separate fill and edge alpha values."""
    num_rectangles = rectangles.shape[0]
    assert fill_colors.shape[0] == num_rectangles, "fill_colors must have the same number of rows as rectangles"
    assert fill_alpha.shape[0] == num_rectangles, "fill_alpha must have the same number of rows as rectangles"
    assert thickness.shape[0] == num_rectangles, "thickness must have the same number of rows as rectangles"
    assert edge_colors.shape[0] == num_rectangles, "edge_colors must have the same number of rows as rectangles"
    assert edge_alpha.shape[0] == num_rectangles, "edge_alpha must have the same number of rows as rectangles"
    assert image.ndim == 3, "image must be a 3D array"
    assert image.shape[2] == 3, "image must have 3 channels (RGB)"
    assert rectangles.ndim == 2, "rectangles must be a 2D array"
    assert rectangles.shape[1] == 4, "rectangles must have 4 columns (x1, y1, x2, y2)"
    assert fill_colors.shape[1] == 3, "fill_colors must have 3 columns (R, G, B)"
    assert edge_colors.shape[1] == 3, "edge_colors must have 3 columns (R, G, B)"
    assert fill_colors.ndim == 2, "fill_colors must be a 2D array"
    assert edge_colors.ndim == 2, "edge_colors must be a 2D array"
    assert fill_alpha.ndim == 1, "fill_alpha must be a 1D array"
    assert edge_alpha.ndim == 1, "edge_alpha must be a 1D array"
    assert thickness.ndim == 1, "thickness must be a 1D array"

    for i in range(rectangles.shape[0]):
        x1, y1, x2, y2 = rectangles[i]

        if x1 > image.shape[1] or x2 <= 0 or y1 > image.shape[0] or y2 <= 0:
            # Skip rectangles that are completely outside the image bounds
            continue

        x1 = min(max(0, x1), image.shape[1])
        y1 = min(max(0, y1), image.shape[0])
        x2 = min(max(0, x2), image.shape[1])
        y2 = min(max(0, y2), image.shape[0])

        if x1 >= x2 or y1 >= y2:
            # Skip rectangles that have no area
            continue

        fill_color = fill_colors[i]
        rect_fill_alpha = fill_alpha[i]
        edge_color = edge_colors[i]
        rect_edge_alpha = edge_alpha[i]
        # fill_color=np.array([0,0,255], dtype=np.uint8)

        # Draw the rectangle fill only if alpha is > 0
        if rect_fill_alpha > 0:
            draw_rectangle_no_edge_jit(
                image=image,
                color=fill_color,
                x1=x1,
                x2=x2,
                y1=y1,
                y2=y2,
                alpha=rect_fill_alpha,
            )

        # Draw the rectangle edges only if thickness > 0 and edge_alpha > 0
        rec_thickness = thickness[i]
        if rec_thickness > 0 and rect_edge_alpha > 0:
            y1b = min(y1 + rec_thickness, y2)
            y2b = max(y2 - rec_thickness, y1)
            x1b = min(x1 + rec_thickness, x2)
            x2b = max(x2 - rec_thickness, x1)
            draw_rectangle_no_edge_jit(
                image=image, color=edge_color, x1=x1, x2=x2, y1=y1, y2=y1b, alpha=rect_edge_alpha
            )
            draw_rectangle_no_edge_jit(
                image=image, color=edge_color, x1=x1, x2=x2, y1=y2b, y2=y2, alpha=rect_edge_alpha
            )
            draw_rectangle_no_edge_jit(
                image=image, color=edge_color, x1=x1, x2=x1b, y1=y1, y2=y2, alpha=rect_edge_alpha
            )
            draw_rectangle_no_edge_jit(
                image=image, color=edge_color, x1=x2b, x2=x2, y1=y1, y2=y2, alpha=rect_edge_alpha
            )


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

        draw_rectangles_jit(
            image=image,
            rectangles=self.corners,
            fill_colors=self.fill_colors,
            fill_alpha=self.fill_alpha,
            edge_colors=self.edge_colors,
            edge_alpha=self.edge_alpha,
            thickness=self.thickness,
        )
        return image


@njit(cache=True, boundscheck=False, nogil=True, fastmath=True)
def draw_line(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: np.ndarray) -> None:
    """Draw a line on an image using Bresenham's algorithm."""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    step_x = 1 if x1 < x2 else -1
    step_y = 1 if y1 < y2 else -1

    # Clip coordinates to image boundaries
    h, w = image.shape[0], image.shape[1]

    if dx > dy:
        # Horizontal-ish line
        err = dx / 2
        while x != x2:
            if 0 <= x < w and 0 <= y < h:
                image[y, x] = color
            err -= dy
            if err < 0:
                y += step_y
                err += dx
            x += step_x
    else:
        # Vertical-ish line
        err = dy / 2
        while y != y2:
            if 0 <= x < w and 0 <= y < h:
                image[y, x] = color
            err -= dx
            if err < 0:
                x += step_x
                err += dy
            y += step_y

    # Draw final point
    if 0 <= x < w and 0 <= y < h:
        image[y, x] = color


@njit(cache=True, boundscheck=False, nogil=True, fastmath=True)
def draw_polyline_jit(image: np.ndarray, x: np.ndarray, y: np.ndarray, color: np.ndarray) -> None:
    """Draw a polyline on an image."""
    assert image.ndim == 3, "image must be a 3D array"
    assert image.shape[2] == 3, "image must have 3 channels (RGB)"
    assert x.ndim == 1, "x must be a 1D array"
    assert y.ndim == 1, "y must be a 1D array"
    assert x.shape[0] == y.shape[0], "x and y must have the same length"
    assert color.ndim == 1 and color.shape[0] == 3, "color must be a 1D array with 3 elements (R, G, B)"
    assert image.shape[0] > 0 and image.shape[1] > 0, "image must have positive height and width"

    for i in range(len(x) - 1):
        x1, y1 = int(x[i]), int(y[i])
        x2, y2 = int(x[i + 1]), int(y[i + 1])
        draw_line(image, x1, y1, x2, y2, color)


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

    draw_polyline_jit(image, x, y, color)
    return image
