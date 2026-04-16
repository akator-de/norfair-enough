"""Miscellaneous helpers: point validation, terminal sizing, warnings."""

import os
from collections.abc import Sequence
from functools import cache
from logging import warning

import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table


def validate_points(points: np.ndarray) -> np.ndarray:
    """Normalize ``points`` to ``(n_points, n_dimensions)`` shape.

    A 1-D array is interpreted as a single point and reshaped to have
    one row; anything with more than two dimensions is rejected via
    :func:`raise_detection_error_message`.

    Parameters
    ----------
    points : np.ndarray
        Array of point coordinates. May be 1-D (single point) or 2-D
        (multiple points).

    Returns
    -------
    np.ndarray
        Array reshaped to ``(n_points, n_dimensions)``.

    Raises
    ------
    ValueError
        If ``points`` has more than two dimensions.
    """
    # If the user is tracking only a single point, reformat it slightly.
    if len(points.shape) == 1:
        points = points[np.newaxis, ...]
    elif len(points.shape) > 2:
        raise_detection_error_message(points)
    return points


def raise_detection_error_message(points):
    """Raise a ``ValueError`` describing a malformed ``Detection.points``.

    Parameters
    ----------
    points : np.ndarray
        The malformed points array whose shape will be included in the
        error message.

    Raises
    ------
    ValueError
        Always raised with a message describing the expected shape and
        a link to the ``Detection`` documentation.
    """
    message = "\n[red]INPUT ERROR:[/red]\n"
    message += f"Each `Detection` object should have a property `points` of shape (n_points, n_dimensions), not {points.shape}. Check your `Detection` list creation code.\n"
    message += "You can read the documentation for the `Detection` class here:\n"
    message += "https://akator-de.github.io/norfair-enough/dev/reference/tracker/#norfair.tracker.Detection\n"
    raise ValueError(message)


def print_objects_as_table(tracked_objects: Sequence):
    """Pretty-print a table summarizing ``tracked_objects`` for debugging.

    Parameters
    ----------
    tracked_objects : Sequence
        Sequence of tracked objects. Each element is expected to have
        ``id``, ``age``, ``hit_counter``, ``last_distance``, and
        ``initializing_id`` attributes (missing attributes are shown
        as ``"?"``).
    """
    print()
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Id", style="yellow", justify="center")
    table.add_column("Age", justify="right")
    table.add_column("Hit Counter", justify="right")
    table.add_column("Last distance", justify="right")
    table.add_column("Init Id", justify="center")
    for obj in tracked_objects:
        last_dist = getattr(obj, "last_distance", None)
        last_dist_str = f"{last_dist:.4f}" if last_dist is not None else "?"
        table.add_row(
            str(getattr(obj, "id", "?")),
            str(getattr(obj, "age", "?")),
            str(getattr(obj, "hit_counter", "?")),
            last_dist_str,
            str(getattr(obj, "initializing_id", "?")),
        )
    console.print(table)


def get_terminal_size(default: tuple[int, int] = (80, 24)) -> tuple[int, int]:
    """Return the terminal ``(columns, lines)``, falling back to ``default``.

    Tries stdin, stdout and stderr in order, returning the first
    successful query.

    Parameters
    ----------
    default : tuple of (int, int), optional
        Fallback ``(columns, lines)`` used when the terminal size
        cannot be determined. Defaults to ``(80, 24)``.

    Returns
    -------
    tuple of (int, int)
        ``(columns, lines)`` of the current terminal.
    """
    columns, lines = default
    for fd in range(0, 3):  # First in order 0=Std In, 1=Std Out, 2=Std Error
        try:
            columns, lines = os.get_terminal_size(fd)
        except OSError:
            continue
        break
    return columns, lines


def get_cutout(points, image):
    """Return the axis-aligned bounding-box cutout of ``points`` in ``image``.

    Parameters
    ----------
    points : np.ndarray
        Array of shape ``(N, 2)`` with x/y coordinates.
    image : np.ndarray
        Image array with at least two spatial dimensions (height, width, ...).

    Returns
    -------
    np.ndarray
        The cropped region.  Returns an empty slice (with a warning) when the
        bounding box is degenerate (zero width or height after clipping).

    Raises
    ------
    ValueError
        If *points* is empty or does not have shape ``(N, 2)``.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("points array is empty")

    image = np.asarray(image)
    if image.ndim < 2:
        raise ValueError(f"image must have at least 2 dimensions, got {image.ndim}")
    img_h, img_w = image.shape[:2]

    min_x = int(np.clip(np.min(points[:, 0]), 0, img_w))
    max_x = int(np.clip(np.max(points[:, 0]), 0, img_w))
    min_y = int(np.clip(np.min(points[:, 1]), 0, img_h))
    max_y = int(np.clip(np.max(points[:, 1]), 0, img_h))

    if min_x == max_x or min_y == max_y:
        warning(
            "get_cutout: degenerate bounding box "
            f"(x={min_x}..{max_x}, y={min_y}..{max_y}); "
            "returning empty cutout"
        )

    return image[min_y:max_y, min_x:max_x]


class DummyOpenCVImport:
    """Placeholder that raises ``ImportError`` when OpenCV is missing.

    Installed as ``cv2`` when the real module cannot be imported, so
    the first attribute access from a video feature raises a clear
    error describing how to install the optional dependency.
    """

    def __getattr__(self, name):
        """Raise ``ImportError`` prompting the user to install OpenCV."""
        raise ImportError(
            r"""[bold red]Missing dependency:[/bold red] You are trying to use Norfair's video features. However, OpenCV is not installed.

Please, make sure there is an existing installation of OpenCV or install Norfair with `pip install norfair-enough\[video]`."""
        )


class DummyMOTMetricsImport:
    """Placeholder that raises ``ImportError`` when ``motmetrics`` is missing.

    Used in the same way as :class:`DummyOpenCVImport` to gate the
    metrics extra.
    """

    def __getattr__(self, name):
        """Raise ``ImportError`` prompting the user to install the metrics extra."""
        raise ImportError(
            r"""[bold red]Missing dependency:[/bold red] You are trying to use Norfair's metrics features without the required dependencies.

Please, install Norfair with `pip install norfair-enough\[metrics]`, or `pip install norfair-enough\[metrics,video]` if you also want video features."""
        )


# lru_cache will prevent re-run the function if the message is the same
@cache
def warn_once(message):
    """Emit ``message`` via ``logging.warning`` at most once per process.

    Parameters
    ----------
    message : str
        The warning text to log. Repeated calls with the same
        ``message`` are suppressed by ``functools.cache``.
    """
    warning(message)
