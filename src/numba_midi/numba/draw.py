"""Draw rectangles and polylines on an image using Numba for performance."""


from numba.core.decorators import njit
import numpy as np


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


@njit(cache=True, boundscheck=False, nogil=True, fastmath=True)
def draw_line(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: np.ndarray) -> None:
    """Draw a line on an image using Bresenham's algorithm."""
    # TODO support float coordinates
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
