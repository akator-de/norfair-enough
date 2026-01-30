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
    """
    Draw the points included in a list of Detections or TrackedObjects.

    Parameters
    ----------
    frame : np.ndarray
        The OpenCV frame to draw on. Modified in place.
    drawables : Union[Sequence[Detection], Sequence[TrackedObject]], optional
        List of objects to draw, Detections and TrackedObjects are accepted.
    radius : Optional[int], optional
        Radius of the circles representing each point.
        By default a sensible value is picked considering the frame size.
    thickness : Optional[int], optional
        Thickness or width of the line.
    color : ColorLike, optional
        This parameter can take:

        1. A color as a tuple of ints describing the BGR `(0, 0, 255)`
        2. A 6-digit hex string `"#FF0000"`
        3. One of the defined color names `"red"`
        4. A string defining the strategy to choose colors from the Palette:

            1. based on the id of the objects `"by_id"`
            2. based on the label of the objects `"by_label"`
            3. random choice `"random"`

        If using `by_id` or `by_label` strategy but your objects don't
        have that field defined (Detections never have ids) the
        selected color will be the same for all objects (Palette's default Color).
    color_by_label : bool, optional
        **Deprecated**. set `color="by_label"`.
    draw_labels : bool, optional
        If set to True, the label is added to a title that is drawn on top of the box.
        If an object doesn't have a label this parameter is ignored.
    draw_scores : bool, optional
        If set to True, the score is added to a title that is drawn on top of the box.
        If an object doesn't have a label this parameter is ignored.
    text_size : Optional[int], optional
        Size of the title, the value is used as a multiplier of the base size of the font.
        By default the size is scaled automatically based on the frame size.
    draw_ids : bool, optional
        If set to True, the id is added to a title that is drawn on top of the box.
        If an object doesn't have an id this parameter is ignored.
    draw_points : bool, optional
        Set to False to hide the points and just draw the text.
    text_thickness : Optional[int], optional
        Thickness of the font. By default it's scaled with the `text_size`.
    text_color : Optional[ColorLike], optional
        Color of the text. By default the same color as the box is used.
    hide_dead_points : bool, optional
        Set this param to False to always draw all points, even the ones considered "dead".
        A point is "dead" when the corresponding value of `TrackedObject.live_points`
        is set to False. If all objects are dead the object is not drawn.
        All points of a detection are considered to be alive.
    detections : Sequence[Detection], optional
        **Deprecated**. use drawables.
    label_size : Optional[int], optional
        **Deprecated**. text_size.

    Returns
    -------
    np.ndarray
        The resulting frame.
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
            position = d.points[d.live_points].mean(axis=0)
            position -= radius
            text = _build_text(
                d, draw_labels=draw_labels, draw_ids=draw_ids, draw_scores=draw_scores
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
    """
    **Deprecated** use [`draw_points`][norfair.drawing.draw_points.draw_points]
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

    _draw_points_alias(
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
