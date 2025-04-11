import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, nogil=True, boundscheck=False)
def draw_rectangles(
    image: np.ndarray,
    rectangles: np.ndarray,
    fill_colors: np.ndarray,
    alpha: np.ndarray,
    thickness: np.ndarray,
    edge_colors: np.ndarray,
):
    """Draw rectangles on an image."""

    for i in range(rectangles.shape[0]):
        x1, y1, x2, y2 = rectangles[i]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        fill_color = fill_colors[i]

        edge_color = edge_colors[i]

        # Draw the rectangle edges
        rec_alpha = alpha[i]
        if rec_alpha == 1:
            image[y1:y2, x1:x2] = fill_color
        else:
            image[y1:y2, x1:x2] = alpha * fill_color + (1 - alpha) * image[y1:y2, x1:x2]

        # Draw the rectangle edges

        rec_thickness = thickness[i]
        if rec_thickness > 0:
            if rec_alpha == 1:
                image[y1 : y1 + rec_thickness, x1:x2] = edge_color
                image[y2 - rec_thickness : y2, x1:x2] = edge_color
                image[y1:y2, x1 : x1 + rec_thickness] = edge_color
                image[y1:y2, x2 - rec_thickness : x2] = edge_color
            else:
                image[y1 : y1 + rec_thickness, x1:x2] = (
                    alpha * edge_color + (1 - alpha) * image[y1 : y1 + rec_thickness, x1:x2]
                )
                image[y2 - rec_thickness : y2, x1:x2] = (
                    alpha * edge_color + (1 - alpha) * image[y2 - rec_thickness : y2, x1:x2]
                )
                image[y1:y2, x1 : x1 + rec_thickness] = (
                    alpha * edge_color + (1 - alpha) * image[y1:y2, x1 : x1 + rec_thickness]
                )
                image[y1:y2, x2 - rec_thickness : x2] = (
                    alpha * edge_color + (1 - alpha) * image[y1:y2, x2 - rec_thickness : x2]
                )
