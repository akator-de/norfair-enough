"""Draw a debugging grid in absolute coordinates over a video frame."""

from functools import lru_cache

import numpy as np

from norfair.camera_motion import CoordinatesTransformation

from .color import Color, ColorType
from .drawer import Drawer


@lru_cache(maxsize=4)
def _get_grid(size, w, h, polar=False):
    """Build a cached grid of points in absolute coordinates.

    The points are sampled so that they lie on the intersection of
    latitude/longitude lines on a unit sphere centered at the camera,
    then projected onto the absolute plane and scaled to fit the frame.
    Results are cached because in absolute coordinates the grid never
    changes between frames.
    """
    # We need to get points on a semi-sphere of radious 1 centered around (0, 0)

    # First step is to get a grid of angles, theta and phi ∈ (-pi/2, pi/2)
    step = np.pi / size
    start = -np.pi / 2 + step / 2
    end = np.pi / 2
    theta, fi = np.mgrid[start:end:step, start:end:step]

    if polar:
        # if polar=True the first frame will show points as if
        # you are on the center of the earth looking at one of the poles.
        # Points on the sphere are defined as [sin(theta) * cos(fi), sin(theta) * sin(fi), cos(theta)]
        # Then we need to intersect the line defined by the point above with the
        # plane z=1 which is the "absolute plane", we do so by dividing by cos(theta), the result becomes
        # [tan(theta) * cos(fi), tan(theta) * sin(phi), 1]
        # note that the z=1 is implied by the coord_transformation so there is no need to add it.
        tan_theta = np.tan(theta)

        X = tan_theta * np.cos(fi)
        Y = tan_theta * np.sin(fi)
    else:
        # otherwhise will show as if you were looking at the equator
        X = np.tan(fi)
        Y = np.divide(np.tan(theta), np.cos(fi))
    # construct the points as x, y coordinates
    points = np.vstack((X.flatten(), Y.flatten())).T
    # scale and center the points
    return points * max(h, w) + np.array([w // 2, h // 2])


def draw_absolute_grid(
    frame: np.ndarray,
    coord_transformations: CoordinatesTransformation,
    grid_size: int = 20,
    radius: int = 2,
    thickness: int = 1,
    color: ColorType = Color.black,
    polar: bool = False,
):
    """Draw a grid of points in absolute coordinates onto ``frame``.

    Useful for debugging camera-motion estimation: the grid stays put
    in world space, so any apparent movement of the points reflects the
    residual error of the estimated transformation.

    The points are drawn as if the camera sat at the center of a unit
    sphere, at the intersection of latitude and longitude lines on that
    sphere's surface.

    Parameters
    ----------
    frame : np.ndarray
        The OpenCV frame to draw on. Modified in place.
    coord_transformations : CoordinatesTransformation
        The coordinate transformation as returned by a
        [`MotionEstimator`][norfair.camera_motion.MotionEstimator].
    grid_size : int, optional
        Number of grid subdivisions per axis.
    radius : int, optional
        Radius (in pixels) of each grid cross.
    thickness : int, optional
        Stroke thickness of each grid cross.
    color : ColorType, optional
        BGR color of the grid crosses.
    polar : bool, optional
        If ``True``, on the first frame the points are drawn as if the
        camera were pointing at a pole. When ``False`` (the default),
        the camera is pointing at the equator.

    """
    h, w, _ = frame.shape

    # get absolute points grid
    points = _get_grid(grid_size, w, h, polar=polar).copy()

    # transform the points to relative coordinates
    if coord_transformations is None:
        points_transformed = points
    else:
        points_transformed = coord_transformations.abs_to_rel(points)

    # filter points that are not visible
    visible_points = points_transformed[
        (points_transformed <= np.array([w, h])).all(axis=1)
        & (points_transformed >= 0).all(axis=1)
    ]
    for point in visible_points:
        Drawer.cross(
            frame, point.astype(int), radius=radius, thickness=thickness, color=color
        )
