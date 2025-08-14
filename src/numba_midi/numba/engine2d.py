"""Functions for 2D geometry operations using Numba."""

from numba.core.decorators import njit
import numpy as np


@njit(cache=True, boundscheck=False, nogil=True, fastmath=True)
def segment_vs_aabb(
    x1: float, y1: float, x2: float, y2: float, left: float, top: float, right: float, bottom: float
) -> bool:
    """
    Check if a line segment intersects with an axis-aligned bounding box (AABB).

    see https://2dengine.com/doc/intersections.html
    """
    # Normalize segment
    dx, dy = x2 - x1, y2 - y1
    d = np.sqrt(dx * dx + dy * dy)
    if d == 0:
        # If the segment is a point, check if it is inside the rectangle
        return left <= x1 <= right and top <= y1 <= bottom

    nx, ny = dx / d, dy / d

    # Minimum and maximum intersection values
    tmin, tmax = 0, d

    # x-axis check
    if nx == 0:
        if x1 < left or x1 > right:
            return False
    else:
        t1, t2 = (left - x1) / nx, (right - x1) / nx
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return False

    # y-axis check
    if ny == 0:
        if y1 < top or y1 > bottom:
            return False
    else:
        t1, t2 = (top - y1) / ny, (bottom - y1) / ny
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return False

    # Two points
    return True


@njit(cache=True, boundscheck=False, nogil=True, fastmath=True)
def rectangles_segment_intersections(rectangles: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """
    Find the intersection points of a set of rectangles with a line segment.

    Parameters
    ----------
    rectangles : np.ndarray
        An array of shape (N, 4) where N is the number of rectangles and each rectangle is defined by its
        top-left corner (x1, y1) and bottom-right corner (x2, y2).
    segment : np.ndarray
        An array of shape (2, 2) where the first row is the start point (x1, y1)
        and the second row is the end point (x2, y2) of the line segment.    Returns:
    -------
    np.ndarray
        A boolean array of shape (M) that is True if the rectangle intersects with the segment and False otherwise.
    """
    # create the boolean array to store the intersection results

    # Unpack the segment points
    x1, y1 = segment[0]
    x2, y2 = segment[1]

    intersections = np.zeros(len(rectangles), dtype=np.bool)

    # Iterate over each rectangle
    for i, rect in enumerate(rectangles):
        left, top, right, bottom = rect
        # Check if the segment intersects with the rectangle
        intersections[i] = segment_vs_aabb(x1, y1, x2, y2, left, top, right, bottom)

    return intersections
