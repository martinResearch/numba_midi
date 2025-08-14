#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint8_t, int32_t
cimport cython

ctypedef cnp.uint8_t DTYPE_uint8_t
ctypedef cnp.int32_t DTYPE_int32_t
ctypedef cnp.float32_t DTYPE_float32_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void draw_rectangle_no_edge_impl(
    cnp.uint8_t[:, :, :] image,
    cnp.uint8_t[:] color, 
    int x1, int x2, int y1, int y2, 
    float alpha
) nogil:
    """Internal implementation for drawing rectangle without edge."""
    cdef int y, x, c
    cdef float inv_alpha = 1.0 - alpha
    
    if alpha == 1.0:
        for y in range(y1, y2):
            for x in range(x1, x2):
                for c in range(3):
                    image[y, x, c] = color[c]
    else:
        for y in range(y1, y2):
            for x in range(x1, x2):
                for c in range(3):
                    image[y, x, c] = <uint8_t>(alpha * color[c] + inv_alpha * image[y, x, c])

def draw_rectangle_no_edge(
    cnp.ndarray[cnp.uint8_t, ndim=3] image,
    cnp.ndarray[cnp.uint8_t, ndim=1] color,
    int x1, int x2, int y1, int y2,
    float alpha
):
    """Draw a rectangle without edge on an image with alpha blending."""
    draw_rectangle_no_edge_impl(image, color, x1, x2, y1, y2, alpha)

def draw_rectangles(
    cnp.ndarray[cnp.uint8_t, ndim=3] image,
    cnp.ndarray[cnp.int32_t, ndim=2] rectangles,
    cnp.ndarray[cnp.uint8_t, ndim=2] fill_colors,
    cnp.ndarray[cnp.float32_t, ndim=1] fill_alpha,
    cnp.ndarray[cnp.int32_t, ndim=1] thickness,
    cnp.ndarray[cnp.uint8_t, ndim=2] edge_colors,
    cnp.ndarray[cnp.float32_t, ndim=1] edge_alpha
):
    """Draw rectangles on an image with separate fill and edge alpha values."""
    cdef int num_rectangles = rectangles.shape[0]
    cdef int img_height = image.shape[0]
    cdef int img_width = image.shape[1]
    cdef int i
    cdef int x1, y1, x2, y2
    cdef int x1b, y1b, x2b, y2b
    cdef int rec_thickness
    cdef float rect_fill_alpha, rect_edge_alpha
    cdef cnp.uint8_t[:] fill_color
    cdef cnp.uint8_t[:] edge_color
    
    # Memory views for better performance
    cdef cnp.uint8_t[:, :, :] image_view = image
    cdef cnp.int32_t[:, :] rectangles_view = rectangles
    cdef cnp.uint8_t[:, :] fill_colors_view = fill_colors
    cdef cnp.float32_t[:] fill_alpha_view = fill_alpha
    cdef cnp.int32_t[:] thickness_view = thickness
    cdef cnp.uint8_t[:, :] edge_colors_view = edge_colors
    cdef cnp.float32_t[:] edge_alpha_view = edge_alpha
    
    for i in range(num_rectangles):
        x1 = rectangles_view[i, 0]
        y1 = rectangles_view[i, 1]
        x2 = rectangles_view[i, 2]
        y2 = rectangles_view[i, 3]
        
        # Skip rectangles that are completely outside the image bounds
        if x1 >= img_width or x2 <= 0 or y1 >= img_height or y2 <= 0:
            continue
            
        # Clip to image bounds
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # Skip rectangles that have no area
        if x1 >= x2 or y1 >= y2:
            continue
            
        fill_color = fill_colors_view[i]
        rect_fill_alpha = fill_alpha_view[i]
        edge_color = edge_colors_view[i]
        rect_edge_alpha = edge_alpha_view[i]
        
        # Draw the rectangle fill only if alpha is > 0
        if rect_fill_alpha > 0:
            draw_rectangle_no_edge_impl(
                image_view, fill_color, x1, x2, y1, y2, rect_fill_alpha
            )
            
        # Draw the rectangle edges only if thickness > 0 and edge_alpha > 0
        rec_thickness = thickness_view[i]
        if rec_thickness > 0 and rect_edge_alpha > 0:
            y1b = min(y1 + rec_thickness, y2)
            y2b = max(y2 - rec_thickness, y1)
            x1b = min(x1 + rec_thickness, x2)
            x2b = max(x2 - rec_thickness, x1)
            
            # Top edge
            draw_rectangle_no_edge_impl(
                image_view, edge_color, x1, x2, y1, y1b, rect_edge_alpha
            )
            # Bottom edge
            draw_rectangle_no_edge_impl(
                image_view, edge_color, x1, x2, y2b, y2, rect_edge_alpha
            )
            # Left edge
            draw_rectangle_no_edge_impl(
                image_view, edge_color, x1, x1b, y1, y2, rect_edge_alpha
            )
            # Right edge
            draw_rectangle_no_edge_impl(
                image_view, edge_color, x2b, x2, y1, y2, rect_edge_alpha
            )

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void draw_line_impl(
    cnp.uint8_t[:, :, :] image,
    int x1, int y1, int x2, int y2,
    cnp.uint8_t[:] color
) nogil:
    """Draw a line on an image using Bresenham's algorithm."""
    cdef int dx = abs(x2 - x1)
    cdef int dy = abs(y2 - y1)
    cdef int x = x1
    cdef int y = y1
    cdef int step_x = 1 if x1 < x2 else -1
    cdef int step_y = 1 if y1 < y2 else -1
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]
    cdef float err
    cdef int c
    
    if dx > dy:
        # Horizontal-ish line
        err = <float>dx / 2.0
        while x != x2:
            if 0 <= x < w and 0 <= y < h:
                for c in range(3):
                    image[y, x, c] = color[c]
            err -= dy
            if err < 0:
                y += step_y
                err += dx
            x += step_x
    else:
        # Vertical-ish line
        err = <float>dy / 2.0
        while y != y2:
            if 0 <= x < w and 0 <= y < h:
                for c in range(3):
                    image[y, x, c] = color[c]
            err -= dx
            if err < 0:
                x += step_x
                err += dy
            y += step_y
    
    # Draw final point
    if 0 <= x < w and 0 <= y < h:
        for c in range(3):
            image[y, x, c] = color[c]

def draw_line(
    cnp.ndarray[cnp.uint8_t, ndim=3] image,
    int x1, int y1, int x2, int y2,
    cnp.ndarray[cnp.uint8_t, ndim=1] color
):
    """Draw a line on an image using Bresenham's algorithm."""
    draw_line_impl(image, x1, y1, x2, y2, color)

def draw_polyline(
    cnp.ndarray[cnp.uint8_t, ndim=3] image,
    cnp.ndarray[cnp.int32_t, ndim=1] x,
    cnp.ndarray[cnp.int32_t, ndim=1] y,
    cnp.ndarray[cnp.uint8_t, ndim=1] color
):
    """Draw a polyline on an image."""
    cdef int i
    cdef int x1, y1, x2, y2
    cdef int n_points = x.shape[0]
    cdef cnp.uint8_t[:, :, :] image_view = image
    cdef cnp.int32_t[:] x_view = x
    cdef cnp.int32_t[:] y_view = y
    cdef cnp.uint8_t[:] color_view = color
    
    for i in range(n_points - 1):
        x1 = x_view[i]
        y1 = y_view[i]
        x2 = x_view[i + 1]
        y2 = y_view[i + 1]
        draw_line_impl(image_view, x1, y1, x2, y2, color_view)
