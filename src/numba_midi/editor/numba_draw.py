"""Draw rectangles and starcaises on an image using Numba for performance."""

from numba import njit
import numpy as np


@njit(cache=True, fastmath=True, nogil=True, boundscheck=False)
def draw_rectangles_jit(
    image: np.ndarray,
    rectangles: np.ndarray,
    fill_colors: np.ndarray,
    alpha: np.ndarray,
    thickness: np.ndarray,
    edge_colors: np.ndarray,
) -> None:
    """Draw rectangles on an image."""
    num_rectangles = rectangles.shape[0]
    assert fill_colors.shape[0] == num_rectangles, "fill_colors must have the same number of rows as rectangles"
    assert alpha.shape[0] == num_rectangles, "alpha must have the same number of rows as rectangles"
    assert thickness.shape[0] == num_rectangles, "thickness must have the same number of rows as rectangles"
    assert edge_colors.shape[0] == num_rectangles, "edge_colors must have the same number of rows as rectangles"
    assert image.ndim == 3, "image must be a 3D array"
    assert image.shape[2] == 3, "image must have 3 channels (RGB)"
    assert rectangles.shape[1] == 4, "rectangles must have 4 columns (x1, y1, x2, y2)"
    assert fill_colors.shape[1] == 3, "fill_colors must have 3 columns (R, G, B)"
    assert edge_colors.shape[1] == 3, "edge_colors must have 3 columns (R, G, B)"
    assert fill_colors.ndim == 2, "fill_colors must be a 2D array"
    assert edge_colors.ndim == 2, "edge_colors must be a 2D array"
    assert alpha.ndim == 1, "alpha must be a 1D array"
    assert thickness.ndim == 1, "thickness must be a 1D array"

    for i in range(rectangles.shape[0]):
        x1, y1, x2, y2 = rectangles[i]

        x1 = min(max(0, x1), image.shape[1])
        y1 = min(max(0, y1), image.shape[0])
        x2 = min(max(0, x2), image.shape[1])
        y2 = min(max(0, y2), image.shape[0])

        fill_color = fill_colors[i]

        edge_color = edge_colors[i]

        # Draw the rectangle
        rec_alpha = alpha[i]
        if rec_alpha == 1:
            image[y1:y2, x1:x2] = fill_color
        else:
            image[y1:y2, x1:x2] = rec_alpha * fill_color + (1 - rec_alpha) * image[y1:y2, x1:x2]

        # Draw the rectangle edges
        rec_thickness = thickness[i]
        if rec_thickness > 0:
            y1b = min(y1 + rec_thickness, y2)
            y2b = max(y2 - rec_thickness, y1)
            x1b = min(x1 + rec_thickness, x2)
            x2b = max(x2 - rec_thickness, x1)
            if rec_alpha == 1:
                image[y1:y1b, x1:x2] = edge_color
                image[y2b:y2, x1:x2] = edge_color
                image[y1b:y2b, x1:x1b] = edge_color
                image[y1b:y2b, x2b:x2] = edge_color
            else:
                image[y1:y1b, x1:x2] = rec_alpha * edge_color + (1 - rec_alpha) * image[y1:y1b, x1:x2]
                image[y2b:y2, x1:x2] = rec_alpha * edge_color + (1 - rec_alpha) * image[y2b:y2, x1:x2]
                image[y1b:y2b, x1:x1b] = rec_alpha * edge_color + (1 - rec_alpha) * image[y1b:y2b, x1:x1b]
                image[y1b:y2b, x2b:x2] = rec_alpha * edge_color + (1 - rec_alpha) * image[y1b:y2b, x2b:x2]


def draw_rectangles(
    image: np.ndarray,
    rectangles: np.ndarray,
    fill_colors: np.ndarray | tuple[int, int, int],
    alpha: np.ndarray | float,
    thickness: np.ndarray | int,
    edge_colors: np.ndarray | tuple[int, int, int] | None,
) -> np.ndarray:
    """Draw rectangles on an image."""
    num_rectangles = rectangles.shape[0]
    if isinstance(alpha, float):
        alpha = np.full(num_rectangles, alpha, dtype=np.float32)
    if isinstance(thickness, int):
        thickness = np.full(num_rectangles, thickness, dtype=np.int32)
    if isinstance(fill_colors, tuple):
        fill_colors = np.array(fill_colors, dtype=np.uint8)
    if edge_colors is None:
        edge_colors = np.zeros_like(fill_colors, dtype=np.uint8)
    if isinstance(edge_colors, tuple):
        edge_colors = np.array(edge_colors, dtype=np.uint8)
    if fill_colors.ndim == 1:
        fill_colors = np.tile(fill_colors, (num_rectangles, 1))
    if edge_colors is None:
        edge_colors = np.zeros_like(fill_colors, dtype=np.uint8)
    if edge_colors.ndim == 1:
        edge_colors = np.tile(edge_colors, (num_rectangles, 1))

    draw_rectangles_jit(image, rectangles, fill_colors, alpha, thickness, edge_colors)
    return image


@njit(cache=True, fastmath=True, nogil=True, boundscheck=False)
def draw_starcaise_jit(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    color: np.ndarray | tuple[int, int, int],
) -> np.ndarray:
    """Draw starcaises on an image.

    x and y are 1D arrays with len(x)==len(y) or len(x)==len(y)+1.

    draw horizontal lines from (x[i], y[i]) to (x[i+1], y[i])
    and vertical lines from (x[i+1], y[i]) to (x[i+1] , y[i+1])
    """
    num_steps = y.shape[0]
    assert x.shape[0] == num_steps or x.shape[0] == num_steps + 1
    assert image.ndim == 3, "image must be a 3D array"
    assert image.shape[2] == 3, "image must have 3 channels (RGB)"
    assert x.ndim == 1, "x must be a 1D array"
    assert y.ndim == 1, "y must be a 1D array"

    for i in range(num_steps - 1):
        x1 = int(x[i])
        y1 = int(y[i])
        x2 = int(x[i + 1])
        y2 = int(y[i + 1])

        if y1 > 0 and y1 < image.shape[0]:
            # Draw horizontal line
            x1b = min(max(0, x1), image.shape[1])
            x2b = min(max(0, x2), image.shape[1])
            image[y1, x1b:x2b, :] = color

        if x2 > 0 and x2 < image.shape[1]:
            # Draw vertical line
            y1b = min(max(0, y1), image.shape[0])
            y2b = min(max(0, y2), image.shape[0])
            ymin = min(y1b, y2b)
            ymax = max(y1b, y2b)
            image[ymin:ymax, x2, :] = color

    # Draw final horizontal line if len(x)==len(y)+1
    if x.shape[0] == num_steps + 1:
        x1 = int(x[num_steps - 1])
        y1 = int(y[num_steps - 1])
        x2 = int(x[num_steps])
        if y1 > 0 and y1 < image.shape[0]:
            # Draw horizontal line
            x1b = min(max(0, x1), image.shape[1])
            x2b = min(max(0, x2), image.shape[1])
            image[y1, x1b:x2b, :] = color

    return image


def draw_starcaise(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    color: np.ndarray | tuple[int, int, int],
) -> np.ndarray:
    """Draw starcaises on an image.

    draw horizontal lines from (x[i], y[i]) to (x[i+1], y[i])
    and vertical lines from (x[i+1], y[i]) to (x[i+1] , y[i+1])
    draw a final horizontalline from (x[i], y[i]) to (width, y[i])
    """
    if isinstance(color, tuple):
        color = np.array(color, dtype=np.uint8)

    draw_starcaise_jit(image, x, y, color)
    return image
