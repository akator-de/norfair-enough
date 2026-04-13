"""Thin wrapper around OpenCV primitives used by the drawing helpers."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from norfair.drawing.color import Color, ColorType
from norfair.tracker import Detection, TrackedObject

try:
    import cv2
except ImportError:
    from norfair.utils import DummyOpenCVImport

    cv2 = DummyOpenCVImport()


class Drawer:
    """Basic drawing primitives used by the higher-level helpers.

    This class encapsulates OpenCV drawing calls behind a stable
    interface so alternative backends can be swapped in without
    touching the rest of the drawing module.
    """

    @classmethod
    def circle(
        cls,
        frame: np.ndarray,
        position: tuple[int, int],
        radius: int | None = None,
        thickness: int | None = None,
        color: ColorType | None = None,
    ) -> np.ndarray:
        """Draw a circle onto ``frame``.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        position : tuple of int
            Center of the circle in pixel coordinates.
        radius : int, optional
            Radius of the circle. By default a sensible value is picked
            based on the frame size.
        thickness : int, optional
            Stroke thickness. By default it is derived from ``radius``.
        color : ColorType, optional
            BGR color. Defaults to black.

        Returns
        -------
        np.ndarray
            The ``frame`` passed in (drawn on in place).

        """
        if radius is None:
            radius = int(max(max(frame.shape) * 0.005, 1))
        if thickness is None:
            thickness = radius - 1
        if color is None:
            color = Color.black

        return cv2.circle(
            frame,
            position,
            radius=radius,
            color=color,
            thickness=thickness,
        )

    @classmethod
    def text(
        cls,
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
        size: float | None = None,
        color: ColorType | None = None,
        thickness: int | None = None,
        shadow: bool = True,
        shadow_color: ColorType = Color.black,
        shadow_offset: int = 1,
    ) -> np.ndarray:
        """Draw text onto ``frame``.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        text : str
            The text to be written.
        position : tuple of int
            Bottom-left corner of the text in pixel coordinates. The
            value is automatically shifted to account for ``thickness``.
        size : float, optional
            Font scale. By default a sensible value is picked based on
            the frame size.
        color : ColorType, optional
            Text color. Defaults to black.
        thickness : int, optional
            Stroke thickness. By default a sensible value is derived
            from ``size``.
        shadow : bool, optional
            If ``True``, a shadow is drawn behind the text to improve
            legibility.
        shadow_color : ColorType, optional
            Color of the shadow.
        shadow_offset : int, optional
            Pixel offset of the shadow relative to the text.

        Returns
        -------
        np.ndarray
            The ``frame`` passed in (drawn on in place).

        """
        font_size = (
            size if size is not None else min(max(max(frame.shape) / 4000, 0.5), 1.5)
        )
        if thickness is None:
            thickness = int(round(font_size) + 1)
        if color is None:
            color = Color.black

        # adjust position based on the thickness
        anchor = (position[0] + thickness // 2, position[1] - thickness // 2)
        if shadow:
            frame = cv2.putText(
                frame,
                text,
                (anchor[0] + shadow_offset, anchor[1] + shadow_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                shadow_color,
                thickness,
                cv2.LINE_AA,
            )
        return cv2.putText(
            frame,
            text,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @classmethod
    def rectangle(
        cls,
        frame: np.ndarray,
        points: Sequence[tuple[int, int]] | np.ndarray,
        color: ColorType | None = None,
        thickness: int | None = None,
    ) -> np.ndarray:
        """Draw a rectangle onto ``frame``.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        points : Sequence[tuple[int, int]] | np.ndarray
            Two opposite corners of the rectangle in the format
            ``[[x0, y0], [x1, y1]]``.
            May be passed as a nested sequence or as a ``(2, 2)`` numpy array —
            both forms are normalised to plain ``(int, int)`` tuples before
            being handed to OpenCV.
        color : ColorType, optional
            BGR outline color. Defaults to black.
        thickness : int, optional
            Stroke thickness of the outline.

        Returns
        -------
        np.ndarray
            The ``frame`` passed in (drawn on in place).

        """
        if color is None:
            color = Color.black
        if thickness is None:
            thickness = 1
        frame = cv2.rectangle(
            frame,
            tuple(points[0]),
            tuple(points[1]),
            color=color,
            thickness=thickness,
        )
        return frame

    @classmethod
    def cross(
        cls,
        frame: np.ndarray,
        center: tuple[int, int],
        radius: int,
        color: ColorType,
        thickness: int,
    ) -> np.ndarray:
        """Draw a ``+``-shaped cross onto ``frame``.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        center : tuple of int
            Center of the cross in pixel coordinates.
        radius : int
            Half-length of each arm of the cross.
        color : ColorType
            BGR color of the lines.
        thickness : int
            Stroke thickness of the lines.

        Returns
        -------
        np.ndarray
            The ``frame`` passed in (drawn on in place).

        """
        middle_x, middle_y = center
        left = center[0] - radius
        top = center[1] - radius
        right = center[0] + radius
        bottom = center[1] + radius
        frame = cls.line(
            frame,
            start=(middle_x, top),
            end=(middle_x, bottom),
            color=color,
            thickness=thickness,
        )
        frame = cls.line(
            frame,
            start=(left, middle_y),
            end=(right, middle_y),
            color=color,
            thickness=thickness,
        )
        return frame

    @classmethod
    def line(
        cls,
        frame: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        color: ColorType = Color.black,
        thickness: int = 1,
    ) -> np.ndarray:
        """Draw a straight line onto ``frame``.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to draw on. Modified in place.
        start : tuple of int
            Starting point in pixel coordinates.
        end : tuple of int
            End point in pixel coordinates.
        color : ColorType, optional
            BGR line color. Defaults to black.
        thickness : int, optional
            Stroke thickness of the line.

        Returns
        -------
        np.ndarray
            The ``frame`` passed in (drawn on in place).

        """
        return cv2.line(
            frame,
            pt1=start,
            pt2=end,
            color=color,
            thickness=thickness,
        )

    @classmethod
    def alpha_blend(
        cls,
        frame1: np.ndarray,
        frame2: np.ndarray,
        alpha: float = 0.5,
        beta: float | None = None,
        gamma: float = 0,
    ) -> np.ndarray:
        """Blend two frames as a weighted sum.

        Parameters
        ----------
        frame1 : np.ndarray
            An OpenCV frame.
        frame2 : np.ndarray
            An OpenCV frame.
        alpha : float, optional
            Weight of ``frame1``.
        beta : float, optional
            Weight of ``frame2``. Defaults to ``1 - alpha``.
        gamma : float, optional
            Scalar added to the weighted sum.

        Returns
        -------
        np.ndarray
            The blended frame.

        """
        if beta is None:
            beta = 1 - alpha
        return cv2.addWeighted(
            src1=frame1, src2=frame2, alpha=alpha, beta=beta, gamma=gamma
        )


class Drawable:
    """Adapter that exposes ``Detection`` and ``TrackedObject`` uniformly.

    The drawing helpers accept either raw ``Detection`` /
    ``TrackedObject`` instances or pre-built ``Drawable`` objects.
    Wrapping an object lets you draw arbitrary point sets with a
    uniform API.

    Parameters
    ----------
    obj : Detection or TrackedObject, optional
        A [Detection][norfair.tracker.Detection] or
        [TrackedObject][norfair.tracker.TrackedObject] used to
        initialize the drawable. If given, all remaining arguments are
        ignored.
    points : np.ndarray, optional
        Point array of shape ``(n_points, n_dimensions)``. Ignored
        when ``obj`` is supplied.
    id : Any, optional
        Object id. Ignored when ``obj`` is supplied.
    label : Any, optional
        Label describing the class of the object. Ignored when ``obj``
        is supplied.
    scores : np.ndarray, optional
        Per-point confidence scores of shape ``(n_points,)``. Ignored
        when ``obj`` is supplied.
    live_points : np.ndarray, optional
        Boolean mask of shape ``(n_points,)`` marking which points are
        still alive. Ignored when ``obj`` is supplied.

    Raises
    ------
    ValueError
        If ``obj`` is not a ``Detection``, ``TrackedObject`` or
        ``None``.

    """

    def __init__(
        self,
        obj: Detection | TrackedObject | None = None,
        points: np.ndarray | None = None,
        id: Any = None,
        label: Any = None,
        scores: np.ndarray | None = None,
        live_points: np.ndarray | None = None,
    ) -> None:
        """Initialize the drawable from ``obj`` or explicit fields."""
        if isinstance(obj, Detection):
            self.points = obj.points
            self.id = None
            self.label = obj.label
            self.scores = obj.scores
            self.live_points = np.ones(obj.points.shape[0]).astype(bool)  # see #13

        elif isinstance(obj, TrackedObject):
            self.points = obj.estimate
            self.id = obj.id
            self.label = obj.label
            self.scores = obj.scores
            self.live_points = obj.live_points
        elif obj is None:
            self.points = points  # pyrefly: ignore[bad-assignment]
            self.id = id
            self.label = label
            self.scores = scores
            self.live_points = live_points  # pyrefly: ignore[bad-assignment]
        else:
            raise ValueError(
                f"Expecting a Detection or a TrackedObject but received {type(obj)}"
            )
