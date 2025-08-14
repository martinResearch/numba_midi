# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

"""
Cython implementations of 2D geometry operations.
Replaces numba_2dengine.py for better performance and precompiled distribution.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

# Initialize numpy
cnp.import_array()

cdef bint _segment_vs_aabb_internal(
    double x1, double y1, double x2, double y2, 
    double left, double top, double right, double bottom
) nogil:
    """
    Check if a line segment intersects with an axis-aligned bounding box (AABB).
    
    see https://2dengine.com/doc/intersections.html
    """
    # Normalize segment
    cdef double dx = x2 - x1
    cdef double dy = y2 - y1
    cdef double d = sqrt(dx * dx + dy * dy)
    
    if d == 0:
        # If the segment is a point, check if it is inside the rectangle
        return left <= x1 <= right and top <= y1 <= bottom

    cdef double nx = dx / d
    cdef double ny = dy / d

    # Minimum and maximum intersection values
    cdef double tmin = 0
    cdef double tmax = d
    cdef double t1, t2

    # x-axis check
    if nx == 0:
        if x1 < left or x1 > right:
            return False
    else:
        t1 = (left - x1) / nx
        t2 = (right - x1) / nx
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
        if tmin > tmax:
            return False

    # y-axis check
    if ny == 0:
        if y1 < top or y1 > bottom:
            return False
    else:
        t1 = (top - y1) / ny
        t2 = (bottom - y1) / ny
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
        if tmin > tmax:
            return False

    return True


def segment_vs_aabb(
    double x1, double y1, double x2, double y2, 
    double left, double top, double right, double bottom
):
    """
    Python wrapper for segment_vs_aabb.
    Check if a line segment intersects with an axis-aligned bounding box (AABB).
    """
    return _segment_vs_aabb_internal(x1, y1, x2, y2, left, top, right, bottom)


def rectangles_segment_intersections(
    cnp.ndarray[cnp.double_t, ndim=2] rectangles,
    cnp.ndarray[cnp.double_t, ndim=2] segment
):
    """
    Find the intersection points of a set of rectangles with a line segment.

    Parameters
    ----------
    rectangles : np.ndarray
        An array of shape (N, 4) where N is the number of rectangles and each rectangle is defined by its
        top-left corner (x1, y1) and bottom-right corner (x2, y2).
    segment : np.ndarray
        An array of shape (2, 2) where the first row is the start point (x1, y1)
        and the second row is the end point (x2, y2) of the line segment.
        
    Returns
    -------
    np.ndarray
        A boolean array of shape (N) that is True if the rectangle intersects with the segment and False otherwise.
    """
    cdef Py_ssize_t n_rectangles = rectangles.shape[0]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] intersections = np.zeros(n_rectangles, dtype=np.bool_)
    
    # Unpack the segment points
    cdef double x1 = segment[0, 0]
    cdef double y1 = segment[0, 1]
    cdef double x2 = segment[1, 0]
    cdef double y2 = segment[1, 1]
    
    cdef Py_ssize_t i
    cdef double left, top, right, bottom
    
    # Iterate over each rectangle
    for i in range(n_rectangles):
        left = rectangles[i, 0]
        top = rectangles[i, 1]
        right = rectangles[i, 2]
        bottom = rectangles[i, 3]
        
        # Check if the segment intersects with the rectangle
        intersections[i] = _segment_vs_aabb_internal(x1, y1, x2, y2, left, top, right, bottom)

    return intersections
