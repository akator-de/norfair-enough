"""Drawers that trace the trajectories of tracked points across frames."""

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np

from norfair.drawing.color import Palette
from norfair.drawing.drawer import Drawer, _safe_int_point
from norfair.tracker import TrackedObject
from norfair.utils import warn_once


class Paths:
    """Draw the trajectories of points of interest on each tracked object.

    Parameters
    ----------
    get_points_to_draw : callable, optional
        Callable taking the ``.estimate`` of a
        [`TrackedObject`][norfair.tracker.TrackedObject] and returning
        a sequence of points whose paths should be drawn. By default
        the mean of all points in the tracker is used.
    thickness : int, optional
        Thickness of the circles representing the path.
    color : tuple of int, optional
        BGR [Color][norfair.drawing.Color] of the path circles. By
        default the color is selected from the active
        [`Palette`][norfair.drawing.Palette] based on the object's id.
    radius : int, optional
        Radius of the circles representing the path.
    attenuation : float, optional
        Value in ``[0, 1]`` controlling how fast existing path pixels
        fade between frames. Use ``0`` to keep the path forever.

    Examples
    --------
    Overlay trajectories on top of tracked objects::

        >>> from norfair import Paths, Tracker, Video
        >>> tracker = Tracker(...)
        >>> path_drawer = Paths()
        >>> with Video(input_path="video.mp4") as video:
        ...     for frame in video:
        ...         detections = get_detections(frame)
        ...         tracked_objects = tracker.update(detections)
        ...         frame = path_drawer.draw(frame, tracked_objects)
        ...         video.write(frame)

    """

    def __init__(
        self,
        get_points_to_draw: Callable[[np.ndarray], np.ndarray] | None = None,
        thickness: int | None = None,
        color: tuple[int, int, int] | None = None,
        radius: int | None = None,
        attenuation: float = 0.01,
    ):
        """Configure the path drawer with its rendering knobs."""
        if get_points_to_draw is None:

            def default_get_points(points):
                return np.array([np.mean(np.array(points), axis=0)])

            self.get_points_to_draw = default_get_points
        else:
            self.get_points_to_draw = (
                get_points_to_draw  # pyrefly: ignore[bad-assignment]
            )

        self.radius = radius
        self.thickness = thickness
        self.color = color
        self.mask: np.ndarray | None = None
        self.attenuation_factor = 1 - attenuation

    def draw(
        self, frame: np.ndarray, tracked_objects: Sequence[TrackedObject]
    ) -> np.ndarray:
        """Update and render the accumulated path mask onto ``frame``.

        !!! warning
            Unlike most other drawers, this method does **not** mutate
            ``frame`` in place — the blended result is returned.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on.
        tracked_objects : Sequence[TrackedObject]
            The [`TrackedObject`][norfair.tracker.TrackedObject] list
            whose points-of-interest are appended to the running path
            mask for this frame.

        Returns
        -------
        np.ndarray
            A new frame with the current path mask blended on top.

        """
        if self.mask is None:
            frame_scale = frame.shape[0] / 100

            if self.radius is None:
                self.radius = int(max(frame_scale * 0.7, 1))
            if self.thickness is None:
                self.thickness = int(max(frame_scale / 7, 1))

            self.mask = np.zeros(frame.shape, np.uint8)

        mask = (self.mask * self.attenuation_factor).astype("uint8")

        for obj in tracked_objects:
            if obj.abs_to_rel is not None:
                warn_once(
                    "It seems that you're using the Path drawer together with MotionEstimator. This is not fully supported and the results will not be what's expected"
                )

            if self.color is None:
                color = Palette.choose_color(obj.id)
            else:
                color = self.color

            points_to_draw = self.get_points_to_draw(obj.estimate)

            for point in points_to_draw:
                safe_pos = _safe_int_point(point)
                if safe_pos is None:
                    continue
                mask = Drawer.circle(
                    mask,
                    position=safe_pos,
                    radius=self.radius,
                    color=color,
                    thickness=self.thickness,
                )

        self.mask = mask
        return Drawer.alpha_blend(mask, frame, alpha=1, beta=1)


class AbsolutePaths:
    """Draw tracked-object trajectories in absolute (world) coordinates.

    Behaves like [`Paths`][norfair.drawing.Paths], but takes camera
    motion into account so trajectories stay pinned to the world
    frame.

    !!! warning
        This drawer is not optimized and can be extremely slow:
        rendering cost grows linearly with
        ``max_history * number_of_tracked_objects``.

    Parameters
    ----------
    get_points_to_draw : callable, optional
        Callable taking the ``.estimate`` of a
        [`TrackedObject`][norfair.tracker.TrackedObject] and returning
        a sequence of points whose paths should be drawn. By default
        the mean of all points in the tracker is used.
    thickness : int, optional
        Thickness of the circles / connecting lines.
    color : tuple of int, optional
        BGR [Color][norfair.drawing.Color] used for every object. By
        default the color is selected from the active palette based on
        the object's id.
    radius : int, optional
        Radius of the circles representing the latest point.
    max_history : int, optional
        Number of past samples to include in the path. Higher values
        make the drawing slower.

    Examples
    --------
    Overlay trajectories on top of tracked objects while accounting
    for camera motion::

        >>> from norfair import AbsolutePaths, MotionEstimator, Tracker, Video
        >>> tracker = Tracker(...)
        >>> motion_estimator = MotionEstimator()
        >>> path_drawer = AbsolutePaths()
        >>> with Video(input_path="video.mp4") as video:
        ...     for frame in video:
        ...         coord_transform = motion_estimator.update(frame)
        ...         detections = get_detections(frame)
        ...         tracked_objects = tracker.update(
        ...             detections, coord_transformations=coord_transform
        ...         )
        ...         frame = path_drawer.draw(frame, tracked_objects, coord_transform)
        ...         video.write(frame)

    """

    def __init__(
        self,
        get_points_to_draw: Callable[[np.ndarray], np.ndarray] | None = None,
        thickness: int | None = None,
        color: tuple[int, int, int] | None = None,
        radius: int | None = None,
        max_history=20,
    ):
        """Configure the absolute-coordinates path drawer."""
        if get_points_to_draw is None:

            def default_get_points(points):
                return np.array([np.mean(np.array(points), axis=0)])

            self.get_points_to_draw = default_get_points
        else:
            self.get_points_to_draw = (
                get_points_to_draw  # pyrefly: ignore[bad-assignment]
            )

        self.radius = radius
        self.thickness = thickness
        self.color = color
        self.past_points: defaultdict[int | None, list[np.ndarray]] = defaultdict(list)
        self.max_history = max_history
        self.alphas = np.linspace(0.99, 0.01, max_history)

    def draw(self, frame, tracked_objects, coord_transform=None):
        """Render accumulated absolute paths onto ``frame``.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        tracked_objects : Sequence[TrackedObject]
            Tracked objects whose absolute trajectories are updated
            and rendered for this frame.
        coord_transform : CoordinatesTransformation, optional
            Transformation between absolute and relative coordinates
            for the current frame. When ``None``, points are assumed
            to already be in the frame's pixel coordinates.

        Returns
        -------
        np.ndarray
            The ``frame`` passed in, with the paths blended on top.

        """
        frame_scale = frame.shape[0] / 100

        if self.radius is None:
            self.radius = int(max(frame_scale * 0.7, 1))
        if self.thickness is None:
            self.thickness = int(max(frame_scale / 7, 1))
        for obj in tracked_objects:
            if not obj.live_points.any():
                continue

            if self.color is None:
                color = Palette.choose_color(obj.id)
            else:
                color = self.color

            points_to_draw = self.get_points_to_draw(obj.get_estimate(absolute=True))

            # Convert from absolute to relative coordinates if transform is provided
            points_rel = (
                coord_transform.abs_to_rel(points_to_draw)
                if coord_transform is not None
                else points_to_draw
            )

            for point in points_rel:
                safe_pos = _safe_int_point(point)
                if safe_pos is None:
                    continue
                Drawer.circle(
                    frame,
                    position=safe_pos,
                    radius=self.radius,
                    color=color,
                    thickness=self.thickness,
                )

            last = points_to_draw
            for i, past_points in enumerate(self.past_points[obj.id]):
                overlay = frame.copy()
                last_rel = (
                    coord_transform.abs_to_rel(last)
                    if coord_transform is not None
                    else last
                )
                past_points_rel = (
                    coord_transform.abs_to_rel(past_points)
                    if coord_transform is not None
                    else past_points
                )
                for j, point in enumerate(past_points_rel):
                    start = _safe_int_point(last_rel[j])
                    end = _safe_int_point(point)
                    if start is None or end is None:
                        continue
                    Drawer.line(
                        overlay,
                        start,
                        end,
                        color=color,
                        thickness=self.thickness,
                    )
                last = past_points

                alpha = self.alphas[i]
                frame = Drawer.alpha_blend(overlay, frame, alpha=alpha)
            # pyrefly: ignore[bad-argument-type]
            self.past_points[obj.id].insert(0, points_to_draw)
            self.past_points[obj.id] = self.past_points[obj.id][: self.max_history]

        # Clean up dead objects to prevent memory leak
        active_ids = {obj.id for obj in tracked_objects}
        dead_ids = [k for k in self.past_points if k not in active_ids]
        for k in dead_ids:
            del self.past_points[k]

        return frame
