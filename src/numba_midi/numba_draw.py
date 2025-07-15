"""Draw rectangles and polylines on an image using Numba for performance."""

from numba.core.decorators import njit
import numpy as np


@njit(cache=True, fastmath=True, nogil=True, boundscheck=False)
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

        # Draw the rectangle fill only if alpha is > 0
        if rect_fill_alpha > 0:
            if rect_fill_alpha == 1:
                image[y1:y2, x1:x2] = fill_color
            else:
                image[y1:y2, x1:x2] = rect_fill_alpha * fill_color + (1 - rect_fill_alpha) * image[y1:y2, x1:x2]

        # Draw the rectangle edges only if thickness > 0 and edge_alpha > 0
        rec_thickness = thickness[i]
        if rec_thickness > 0 and rect_edge_alpha > 0:
            y1b = min(y1 + rec_thickness, y2)
            y2b = max(y2 - rec_thickness, y1)
            x1b = min(x1 + rec_thickness, x2)
            x2b = max(x2 - rec_thickness, x1)
            if rect_edge_alpha == 1:
                image[y1:y1b, x1:x2] = edge_color
                image[y2b:y2, x1:x2] = edge_color
                image[y1b:y2b, x1:x1b] = edge_color
                image[y1b:y2b, x2b:x2] = edge_color
            else:
                image[y1:y1b, x1:x2] = rect_edge_alpha * edge_color + (1 - rect_edge_alpha) * image[y1:y1b, x1:x2]
                image[y2b:y2, x1:x2] = rect_edge_alpha * edge_color + (1 - rect_edge_alpha) * image[y2b:y2, x1:x2]
                image[y1b:y2b, x1:x1b] = rect_edge_alpha * edge_color + (1 - rect_edge_alpha) * image[y1b:y2b, x1:x1b]
                image[y1b:y2b, x2b:x2] = rect_edge_alpha * edge_color + (1 - rect_edge_alpha) * image[y1b:y2b, x2b:x2]


def draw_rectangles(
    image: np.ndarray,
    rectangles: np.ndarray,
    fill_colors: np.ndarray | tuple[int, int, int],
    alpha: np.ndarray | float = 1.0,
    edge_colors: np.ndarray | tuple[int, int, int] | None = None,
    thickness: np.ndarray | int = 0,
    fill_alpha: np.ndarray | float | None = None,
    edge_alpha: np.ndarray | float | None = None,
    prefilter: bool = True,
) -> np.ndarray:
    """Draw rectangles on an image with separate fill and edge alpha values.

    Args:
        image: The image to draw on, shape (height, width, 3)
        rectangles: The rectangles to draw, shape (n, 4) where each row is (x1, y1, x2, y2)
        fill_colors: The fill colors, shape (n, 3) or a single (r, g, b) tuple
        alpha: The alpha value for both fill and edge (legacy parameter, use fill_alpha/edge_alpha)
        edge_colors: The edge colors, shape (n, 3) or a single (r, g, b) tuple
        thickness: The edge thickness, shape (n,) or a single int
        fill_alpha: The fill alpha value, shape (n,) or a single float, overrides alpha
        edge_alpha: The edge alpha value, shape (n,) or a single float, overrides alpha
        prefilter: If True, filter out rectangles that are completely outside the image bounds
    Returns:
        The image with rectangles drawn
    """
    num_rectangles = rectangles.shape[0]

    if prefilter:
        # filter out rectangles that are completely outside the image bounds
        keep_mask = (
            (rectangles[:, 0] < image.shape[1])
            & (rectangles[:, 2] > 0)
            & (rectangles[:, 1] < image.shape[0])
            & (rectangles[:, 3] > 0)
        )
        rectangles = rectangles[keep_mask]
        if isinstance(fill_colors, np.ndarray):
            fill_colors = fill_colors[keep_mask]
        if isinstance(edge_colors, np.ndarray):
            edge_colors = edge_colors[keep_mask]
        if isinstance(fill_alpha, np.ndarray):
            fill_alpha = fill_alpha[keep_mask]
        if isinstance(edge_alpha, np.ndarray):
            edge_alpha = edge_alpha[keep_mask]
        if isinstance(thickness, np.ndarray):
            thickness = thickness[keep_mask]
        if isinstance(alpha, np.ndarray):
            alpha = alpha[keep_mask]
        if isinstance(fill_alpha, np.ndarray):
            fill_alpha = fill_alpha[keep_mask]
        if isinstance(edge_alpha, np.ndarray):
            edge_alpha = edge_alpha[keep_mask]
        num_rectangles = rectangles.shape[0]
    if num_rectangles == 0:
        # If no rectangles to draw, return the original image
        return image
    # Handle the legacy alpha parameter for backward compatibility
    if fill_alpha is None:
        fill_alpha = alpha
    if edge_alpha is None:
        edge_alpha = alpha

    # Convert single values to arrays
    if isinstance(fill_alpha, float):
        fill_alpha = np.full(num_rectangles, fill_alpha, dtype=np.float32)
    if isinstance(edge_alpha, float):
        edge_alpha = np.full(num_rectangles, edge_alpha, dtype=np.float32)
    if isinstance(thickness, int):
        thickness = np.full(num_rectangles, thickness, dtype=np.int32)

    # Convert single color values to arrays
    if isinstance(fill_colors, tuple):
        fill_colors = np.array(fill_colors, dtype=np.uint8)
    if fill_colors.ndim == 1:
        fill_colors = np.tile(fill_colors, (num_rectangles, 1))

    # Handle edge colors
    if edge_colors is None:
        edge_colors = np.zeros_like(fill_colors, dtype=np.uint8)
    if isinstance(edge_colors, tuple):
        edge_colors = np.array(edge_colors, dtype=np.uint8)
    if edge_colors.ndim == 1:
        edge_colors = np.tile(edge_colors, (num_rectangles, 1))

    if fill_colors.shape[0] != num_rectangles:
        raise ValueError("fill_colors must have the same number of rows as rectangles")
    if fill_alpha.shape[0] != num_rectangles:
        raise ValueError("fill_alpha must have the same number of rows as rectangles")
    if thickness.shape[0] != num_rectangles:
        raise ValueError("thickness must have the same number of rows as rectangles")
    if edge_colors.shape[0] != num_rectangles:
        raise ValueError("edge_colors must have the same number of rows as rectangles")
    if edge_alpha.shape[0] != num_rectangles:
        raise ValueError("edge_alpha must have the same number of rows as rectangles")
    if image.ndim != 3:
        raise ValueError("image must be a 3D array")
    if image.shape[2] != 3:
        raise ValueError("image must have 3 channels (RGB)")
    if rectangles.ndim != 2:
        raise ValueError("rectangles must be a 2D array")
    if rectangles.shape[1] != 4:
        raise ValueError("rectangles must have 4 columns (x1, y1, x2, y2)")
    if fill_colors.shape[1] != 3:
        raise ValueError("fill_colors must have 3 columns (R, G, B)")
    if edge_colors.shape[1] != 3:
        raise ValueError("edge_colors must have 3 columns (R, G, B)")
    if fill_colors.ndim != 2:
        raise ValueError("fill_colors must be a 2D array")
    if edge_colors.ndim != 2:
        raise ValueError("edge_colors must be a 2D array")
    if fill_alpha.ndim != 1:
        raise ValueError("fill_alpha must be a 1D array")
    if edge_alpha.ndim != 1:
        raise ValueError("edge_alpha must be a 1D array")
    if thickness.ndim != 1:
        raise ValueError("thickness must be a 1D array")

    draw_rectangles_jit(image, rectangles, fill_colors, fill_alpha, thickness, edge_colors, edge_alpha)
    return image


@njit(cache=True, fastmath=True, nogil=True)
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


@njit(cache=True, fastmath=True, nogil=True)
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
