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
    """
    # If the user is tracking only a single point, reformat it slightly.
    if len(points.shape) == 1:
        points = points[np.newaxis, ...]
    elif len(points.shape) > 2:
        raise_detection_error_message(points)
    return points


def raise_detection_error_message(points):
    """Raise a ``ValueError`` describing a malformed ``Detection.points``."""
    message = "\n[red]INPUT ERROR:[/red]\n"
    message += f"Each `Detection` object should have a property `points` of shape (n_points, n_dimensions), not {points.shape}. Check your `Detection` list creation code.\n"
    message += "You can read the documentation for the `Detection` class here:\n"
    message += "https://akator-de.github.io/norfair-enough/latest/reference/tracker/#norfair.tracker.Detection\n"
    raise ValueError(message)


def print_objects_as_table(tracked_objects: Sequence):
    """Pretty-print a table summarizing ``tracked_objects`` for debugging."""
    print()
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Id", style="yellow", justify="center")
    table.add_column("Age", justify="right")
    table.add_column("Hit Counter", justify="right")
    table.add_column("Last distance", justify="right")
    table.add_column("Init Id", justify="center")
    for obj in tracked_objects:
        table.add_row(
            str(obj.id),
            str(obj.age),
            str(obj.hit_counter),
            f"{obj.last_distance:.4f}",
            str(obj.initializing_id),
        )
    console.print(table)


def get_terminal_size(default: tuple[int, int] = (80, 24)) -> tuple[int, int]:
    """Return the terminal ``(columns, lines)``, falling back to ``default``.

    Tries stdin, stdout and stderr in order, returning the first
    successful query.
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
    """Return the axis-aligned bounding-box cutout of ``points`` in ``image``."""
    max_x = int(max(points[:, 0]))
    min_x = int(min(points[:, 0]))
    max_y = int(max(points[:, 1]))
    min_y = int(min(points[:, 1]))
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
    """Emit ``message`` via ``logging.warning`` at most once per process."""
    warning(message)
