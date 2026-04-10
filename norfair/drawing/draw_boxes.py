"""Draw detection / tracked-object bounding boxes onto a video frame."""

from collections.abc import Sequence

import numpy as np

from norfair.tracker import Detection, TrackedObject
from norfair.utils import warn_once

from .color import ColorLike, Palette, parse_color
from .drawer import Drawable, Drawer
from .utils import _build_text


def draw_boxes(
    frame: np.ndarray,
    drawables: Sequence[Detection] | Sequence[TrackedObject] | None = None,
    color: ColorLike = "by_id",
    thickness: int | None = None,
    random_color: bool | None = None,  # Deprecated
    color_by_label: bool | None = None,  # Deprecated
    draw_labels: bool = False,
    text_size: float | None = None,
    draw_ids: bool = True,
    text_color: ColorLike | None = None,
    text_thickness: int | None = None,
    draw_box: bool = True,
    detections: Sequence["Detection"] | None = None,  # Deprecated
    line_color: ColorLike | None = None,  # Deprecated
    line_width: int | None = None,  # Deprecated
    label_size: int | None = None,  # Deprecated
    draw_scores: bool = False,
) -> np.ndarray:
    """Draw bounding boxes for ``Detection`` or ``TrackedObject``.

    Parameters
    ----------
    frame : np.ndarray
        The OpenCV frame to draw on. Modified in place.
    drawables : Sequence[Detection] or Sequence[TrackedObject], optional
        Objects to draw. Each object is assumed to contain two
        two-dimensional points defining the bounding box as
        ``[[x0, y0], [x1, y1]]``.
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
    thickness : int, optional
        Thickness (width) of the box outline.
    random_color : bool, optional
        **Deprecated.** Set ``color="random"`` instead.
    color_by_label : bool, optional
        **Deprecated.** Set ``color="by_label"`` instead.
    draw_labels : bool, optional
        If ``True``, the label is drawn above the box. Ignored when the
        object has no label.
    draw_scores : bool, optional
        If ``True``, the detection score is drawn above the box. Ignored
        when the object has no score.
    text_size : float, optional
        Size multiplier for the base font used for the text. By default,
        the size is scaled automatically based on the frame size.
    draw_ids : bool, optional
        If ``True``, the id is drawn above the box. Ignored when the
        object has no id.
    text_color : ColorLike, optional
        Color of the text. Defaults to the same color as the box.
    text_thickness : int, optional
        Stroke thickness of the text. By default it scales with
        ``text_size``.
    draw_box : bool, optional
        Set to ``False`` to hide the box and only draw the text.
    detections : Sequence[Detection], optional
        **Deprecated.** Use ``drawables``.
    line_color : ColorLike, optional
        **Deprecated.** Use ``color``.
    line_width : int, optional
        **Deprecated.** Use ``thickness``.
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
    if random_color is not None:
        warn_once(
            'Parameter "random_color" is deprecated, set `color="random"` instead'
        )
        color = "random"
    if color_by_label is not None:
        warn_once(
            'Parameter "color_by_label" is deprecated, set `color="by_label"` instead'
        )
        color = "by_label"
    if detections is not None:
        warn_once('Parameter "detections" is deprecated, use "drawables" instead')
        drawables = detections
    if line_color is not None:
        warn_once('Parameter "line_color" is deprecated, use "color" instead')
        color = line_color
    if line_width is not None:
        warn_once('Parameter "line_width" is deprecated, use "thickness" instead')
        thickness = line_width
    if label_size is not None:
        warn_once('Parameter "label_size" is deprecated, use "text_size" instead')
        text_size = label_size
    # end

    if color is None:
        color = "by_id"
    if thickness is None:
        thickness = int(max(frame.shape) / 500)

    if drawables is None:
        return frame

    if text_color is not None:
        text_color = parse_color(text_color)

    for obj in drawables:
        if not isinstance(obj, Drawable):
            d = Drawable(obj)
        else:
            d = obj

        if color == "by_id":
            obj_color = Palette.choose_color(d.id)
        elif color == "by_label":
            obj_color = Palette.choose_color(d.label)
        elif color == "random":
            obj_color = Palette.choose_color(np.random.rand())
        else:
            obj_color = parse_color(color)

        points = d.points.astype(int)
        if draw_box:
            Drawer.rectangle(
                frame,
                tuple(points),
                color=obj_color,
                thickness=thickness,
            )

        text = _build_text(
            d, draw_labels=draw_labels, draw_ids=draw_ids, draw_scores=draw_scores
        )
        if text:
            if text_color is None:
                obj_text_color = obj_color
            else:
                obj_text_color = text_color
            # the anchor will become the bottom-left of the text,
            # we select-top left of the bbox compensating for the thickness of the box
            text_anchor = (
                points[0, 0] - thickness // 2,
                points[0, 1] - thickness // 2 - 1,
            )
            frame = Drawer.text(
                frame,
                text,
                position=text_anchor,
                size=text_size,
                color=obj_text_color,
                thickness=text_thickness,
            )

    return frame


def draw_tracked_boxes(
    frame: np.ndarray,
    objects: Sequence["TrackedObject"],
    border_colors: tuple[int, int, int] | None = None,
    border_width: int | None = None,
    id_size: int | None = None,
    id_thickness: int | None = None,
    draw_box: bool = True,
    color_by_label: bool = False,
    draw_labels: bool = False,
    label_size: int | None = None,
    label_width: int | None = None,
) -> np.ndarray:
    """Draw tracked-object bounding boxes onto ``frame``.

    .. deprecated::
        Use :func:`draw_boxes` instead. This function is kept for
        backward compatibility and forwards its arguments to
        :func:`draw_boxes`.

    Parameters
    ----------
    frame : np.ndarray
        The OpenCV frame to draw on. Modified in place.
    objects : Sequence[TrackedObject]
        The tracked objects to draw.
    border_colors : tuple of int, optional
        BGR border color. Ignored if ``color_by_label`` is ``True``.
    border_width : int, optional
        Thickness of the border line.
    id_size : int, optional
        Size of the id text. Set to ``0`` to disable id rendering.
    id_thickness : int, optional
        Stroke thickness of the id text.
    draw_box : bool, optional
        Set to ``False`` to hide the box and only draw the text.
    color_by_label : bool, optional
        If ``True``, color objects by label instead of by a fixed
        ``border_colors``.
    draw_labels : bool, optional
        If ``True``, draw the label above the box.
    label_size : int, optional
        Size of the label text.
    label_width : int, optional
        Stroke thickness of the label text.

    Returns
    -------
    np.ndarray
        The ``frame`` passed in (drawn on in place).

    """
    warn_once("draw_tracked_boxes is deprecated, use draw_boxes instead")
    # Determine color - default to "by_id" if border_colors is None
    selected_color: ColorLike = (
        "by_label"
        if color_by_label
        else (border_colors if border_colors is not None else "by_id")
    )
    return draw_boxes(
        frame=frame,
        drawables=objects,
        color=selected_color,
        thickness=border_width,
        text_size=label_size or id_size,
        text_thickness=id_thickness or label_width,
        draw_labels=draw_labels,
        draw_ids=id_size is not None and id_size > 0,
        draw_box=draw_box,
    )
