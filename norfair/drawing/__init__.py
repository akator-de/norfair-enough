"""Collection of drawing helpers: detections, tracked objects, paths and overlays."""

from .absolute_grid import draw_absolute_grid
from .color import Color, ColorLike, ColorType, Palette
from .draw_boxes import draw_boxes, draw_tracked_boxes
from .draw_points import draw_points, draw_tracked_objects
from .drawer import Drawable
from .fixed_camera import FixedCamera
from .path import AbsolutePaths, Paths

__all__ = [
    "draw_absolute_grid",
    "Color",
    "ColorLike",
    "ColorType",
    "Palette",
    "draw_boxes",
    "draw_tracked_boxes",
    "draw_points",
    "draw_tracked_objects",
    "Drawable",
    "FixedCamera",
    "AbsolutePaths",
    "Paths",
]
