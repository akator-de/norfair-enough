"""Draw detection / tracked-object points onto a video frame."""

from collections.abc import Sequence

import numpy as np

from norfair.tracker import Detection, TrackedObject
from norfair.utils import warn_once

from .color import ColorLike, Palette, parse_color
from .drawer import Drawable, Drawer
from .utils import _build_text


def draw_points(
    frame: np.ndarray,
    drawables: Sequence[Detection] | Sequence[TrackedObject] | None = None,
    radius: int | None = None,
    thickness: int | None = None,
    color: ColorLike = "by_id",
    color_by_label: bool | None = None,  # deprecated
    draw_labels: bool = True,
    text_size: int | None = None,
    draw_ids: bool = True,
    draw_points: bool = True,  # pylint: disable=redefined-outer-name
    text_thickness: int | None = None,
    text_color: ColorLike | None = None,
    hide_dead_points: bool = True,
    detections: Sequence["Detection"] | None = None,  # deprecated
    label_size: int | None = None,  # deprecated
    draw_scores: bool = False,
) -> np.ndarray:
    """Draw the points of a list of ``Detection`` or ``TrackedObject``.

    Parameters
    ----------
    frame : np.ndarray
        The OpenCV frame to draw on. Modified in place.
    drawables : Sequence[Detection] or Sequence[TrackedObject], optional
        Objects to draw. Both ``Detection`` and ``TrackedObject`` are
        accepted.
    radius : int, optional
        Radius of the circles representing each point. By default a
        sensible value is picked based on the frame size.
    thickness : int, optional
        Thickness of the stroke. ``-1`` (the default) produces filled
        circles.
    color : ColorLike, optional
        The color to use. May be:

        1. A BGR int tuple like ``(0, 0, 255)``.
        2. A 6-digit hex string such as ``"#FF0000"``.
        3. One of the predefined color names (e.g. ``"red"``).
        4. A palette strategy — ``"by_id"``, ``"by_label"`` or
           ``"random"``.

        When ``"by_id"`` or ``"by_label"`` is used but the object lacks
        that field (detections never have ``id``), every object is drawn
        in the palette's default color.
    color_by_label : bool, optional
        **Deprecated.** Set ``color="by_label"`` instead.
    draw_labels : bool, optional
        If ``True``, the label is drawn above the points. Ignored when
        the object has no label.
    draw_scores : bool, optional
        If ``True``, the detection score is drawn above the points.
        Ignored when the object has no score.
    text_size : int, optional
        Size multiplier for the base font used for the text. By default,
        the size is scaled automatically based on the frame size.
    draw_ids : bool, optional
        If ``True``, the id is drawn above the points. Ignored when the
        object has no id.
    draw_points : bool, optional
        Set to ``False`` to hide the points and only draw the text.
    text_thickness : int, optional
        Stroke thickness of the text. By default it scales with
        ``text_size``.
    text_color : ColorLike, optional
        Color of the text. Defaults to the object's color.
    hide_dead_points : bool, optional
        Set to ``False`` to draw all points, including "dead" ones. A
        point is dead when the corresponding entry in
        ``TrackedObject.live_points`` is ``False``. If every point is
        dead the whole object is skipped. Detection points are always
        treated as live.
    detections : Sequence[Detection], optional
        **Deprecated.** Use ``drawables``.
    label_size : int, optional
        **Deprecated.** Use ``text_size``.

    Returns
    -------
    np.ndarray
        The ``frame`` passed in (drawn on in place).
    """
    #
    # handle deprecated parameters
    #
    if color_by_label is not None:
        warn_once(
            'Parameter "color_by_label" on function draw_points is deprecated, set `color="by_label"` instead'
        )
        color = "by_label"
    if detections is not None:
        warn_once(
            "Parameter 'detections' on function draw_points is deprecated, use 'drawables' instead"
        )
        drawables = detections
    if label_size is not None:
        warn_once(
            "Parameter 'label_size' on function draw_points is deprecated, use 'text_size' instead"
        )
        text_size = label_size
    # end

    if drawables is None:
        return frame

    if text_color is not None:
        text_color = parse_color(text_color)

    if color is None:
        color = "by_id"
    if thickness is None:
        thickness = -1
    if radius is None:
        radius = int(round(max(max(frame.shape) * 0.002, 1)))

    for o in drawables:
        if not isinstance(o, Drawable):
            d = Drawable(o)
        else:
            d = o

        if hide_dead_points and not d.live_points.any():
            continue

        if color == "by_id":
            obj_color = Palette.choose_color(d.id)
        elif color == "by_label":
            obj_color = Palette.choose_color(d.label)
        elif color == "random":
            obj_color = Palette.choose_color(np.random.rand())
        else:
            obj_color = parse_color(color)

        if text_color is None:
            obj_text_color = obj_color
        else:
            obj_text_color = text_color

        if draw_points:
            for point, live in zip(d.points, d.live_points):
                if live or not hide_dead_points:
                    Drawer.circle(
                        frame,
                        tuple(point.astype(int)),  # pyrefly: ignore[bad-argument-type]
                        radius=radius,
                        color=obj_color,
                        thickness=thickness,
                    )

        if draw_labels or draw_ids or draw_scores:
            live = d.points[d.live_points]
            if len(live) > 0:
                position = live.mean(axis=0)
                position -= radius
                text = _build_text(
                    d,
                    draw_labels=draw_labels,
                    draw_ids=draw_ids,
                    draw_scores=draw_scores,
                )

                Drawer.text(
                    frame,
                    text,
                    tuple(position.astype(int)),  # pyrefly: ignore[bad-argument-type]
                    size=text_size,
                    color=obj_text_color,
                    thickness=text_thickness,
                )

    return frame


# Alias needed because draw_tracked_objects has a parameter named "draw_points"
# which shadows the function name in its local scope.
_draw_points_alias = draw_points


def draw_tracked_objects(
    frame: np.ndarray,
    objects: Sequence["TrackedObject"],
    radius: int | None = None,
    color: ColorLike | None = None,
    id_size: float | None = None,
    id_thickness: int | None = None,
    draw_points: bool = True,  # pylint: disable=redefined-outer-name
    color_by_label: bool = False,
    draw_labels: bool = False,
    label_size: int | None = None,
):
    """Draw tracked objects onto ``frame``.

    .. deprecated::
        Use :func:`draw_points` instead. This function is kept for
        backward compatibility and forwards its arguments to
        :func:`draw_points`.

    Parameters
    ----------
    frame : np.ndarray
        The OpenCV frame to draw on. Modified in place.
    objects : Sequence[TrackedObject]
        The tracked objects to draw.
    radius : int, optional
        Radius of the circles representing each point.
    color : ColorLike, optional
        Color to use. See :func:`draw_points` for accepted formats.
    id_size : float, optional
        Size multiplier for the id text. Set to ``0`` to disable id
        rendering.
    id_thickness : int, optional
        Stroke thickness of the id text.
    draw_points : bool, optional
        Set to ``False`` to hide the point circles and only draw text.
    color_by_label : bool, optional
        If ``True``, color objects by label instead of id.
    draw_labels : bool, optional
        If ``True``, draw the label above the points.
    label_size : int, optional
        Size of the label text.

    Returns
    -------
    np.ndarray
        The ``frame`` passed in (drawn on in place).

    """
    warn_once("draw_tracked_objects is deprecated, use draw_points instead")

    frame_scale = frame.shape[0] / 100
    if radius is None:
        radius = int(frame_scale * 0.5)
    if id_size is None:
        id_size = frame_scale / 10
    if id_thickness is None:
        id_thickness = int(frame_scale / 5)
    if label_size is None:
        label_size = int(max(frame_scale / 100, 1))

    # Determine color - default to "by_id" if None
    selected_color: ColorLike = (
        "by_label" if color_by_label else (color if color is not None else "by_id")
    )

    # Convert id_size to int if it's a float
    text_size_value: int | None = None
    if label_size is not None:
        text_size_value = label_size
    elif id_size is not None:
        text_size_value = int(id_size)

    return _draw_points_alias(
        frame=frame,
        drawables=objects,
        color=selected_color,
        radius=radius,
        thickness=None,
        draw_labels=draw_labels,
        draw_ids=id_size is not None and id_size > 0,
        draw_points=draw_points,
        text_size=text_size_value,
        text_thickness=id_thickness,
        text_color=None,
        hide_dead_points=True,
    )
