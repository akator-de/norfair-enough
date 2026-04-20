"""Video I/O helpers: frame iteration, display, and output writing."""

import contextlib
import os
import time
from typing import TYPE_CHECKING, Any

try:
    import cv2
except ImportError:
    from .utils import DummyOpenCVImport

    cv2 = DummyOpenCVImport()  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    import cv2 as cv2_types
else:
    cv2_types = None  # type: ignore[misc]

import numpy as np
from rich import print
from rich.progress import BarColumn, Progress, ProgressColumn, TimeRemainingColumn

from norfair import metrics

from .utils import get_terminal_size


def _get_fourcc(codec: str) -> int:
    """Return the integer fourcc code for ``codec``."""
    return cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]


# Cap the progress-bar refresh rate at ~20 Hz: Rich rendering is a
# meaningful share of per-frame cost on fast pipelines.
_PROGRESS_REFRESH_INTERVAL_S = 0.05


class Video:
    """Simple pythonic wrapper around an OpenCV video source and sink.

    Yields raw OpenCV frames so the full OpenCV toolbox is available
    for frame-level processing, while taking care of capture setup,
    progress-bar rendering and writer initialization.

    Parameters
    ----------
    camera : int, optional
        Device id of the camera to read from. Webcams typically use
        ``0``. Mutually exclusive with ``input_path``.
    input_path : str, optional
        Path to the video file to read. Mutually exclusive with
        ``camera``.
    output_path : str, optional
        Where the output video should be written. May be a directory
        (an output filename is then auto-generated) or a full file
        path.
    output_fps : float, optional
        Frames per second for the output file. Defaults to the input
        source's fps. This is useful with live cameras when the
        effective processing rate is lower than the camera's native
        rate.
    label : str, optional
        Optional label appended to the progress-bar description.
    output_fourcc : str, optional
        OpenCV codec for the output file. Defaults to ``"mp4v"`` for
        ``.mp4`` outputs and ``"XVID"`` for ``.avi``. Try ``"avc1"``
        or ``"H264"`` for smaller files when available.
    output_extension : str, optional
        File extension used when ``output_path`` is a directory.

    Examples
    --------
    Read, process and write a video::

        >>> with Video(input_path="video.mp4") as video:
        ...     for frame in video:
        ...         # << your frame processing goes here >>
        ...         video.write(frame)

    The context manager releases the underlying ``VideoCapture`` and
    ``VideoWriter`` even if the loop is interrupted. Iterating without
    ``with`` also works — resources are freed when the iterator is
    exhausted or when :meth:`close` is called.

    Raises
    ------
    ValueError
        If neither or both of ``camera`` / ``input_path`` are
        supplied, or if ``camera`` is not an ``int``.

    """

    def __init__(
        self,
        camera: int | None = None,
        input_path: str | None = None,
        output_path: str = ".",
        output_fps: float | None = None,
        label: str = "",
        output_fourcc: str | None = None,
        output_extension: str = "mp4",
    ):
        """Open the capture and set up the progress bar."""
        self.camera = camera
        self.input_path = input_path
        self.output_path = output_path
        self.label = label
        self.output_fourcc = output_fourcc
        self.output_extension = output_extension
        self.output_video: Any = None  # cv2.VideoWriter or None

        # Input validation
        if (input_path is None and camera is None) or (
            input_path is not None and camera is not None
        ):
            raise ValueError(
                "You must set either 'camera' or 'input_path' arguments when setting 'Video' class"
            )
        if camera is not None and type(camera) is not int:
            raise ValueError(
                "Argument 'camera' refers to the device-id of your camera, and must be an int. Setting it to 0 usually works if you don't know the id."
            )

        # Read Input Video
        if self.input_path is not None:
            if "~" in self.input_path:
                self.input_path = os.path.expanduser(self.input_path)
            if not os.path.isfile(self.input_path):
                self._fail(
                    f"[bold red]Error:[/bold red] File '{self.input_path}' does not exist."
                )
            self.video_capture = cv2.VideoCapture(self.input_path)
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                self._fail(
                    f"[bold red]Error:[/bold red] '{self.input_path}' does not seem to be a video file supported by OpenCV. If the video file is not the problem, please check that your OpenCV installation is working correctly."
                )
            description = os.path.basename(self.input_path)
        else:
            assert self.camera is not None  # Validated in input_validation above
            self.video_capture = cv2.VideoCapture(self.camera)
            total_frames = 0
            description = f"Camera({self.camera})"
        self.output_fps = (
            output_fps
            if output_fps is not None
            else self.video_capture.get(cv2.CAP_PROP_FPS)
        )
        self.input_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.input_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_counter = 0
        self._closed = False

        # Setup progressbar
        if self.label:
            description += f" | {self.label}"
        progress_bar_fields: list[str | ProgressColumn] = [
            "[progress.description]{task.description}",
            BarColumn(),
            "[yellow]{task.fields[process_fps]:.2f}fps[/yellow]",
        ]
        if self.input_path is not None:
            progress_bar_fields.insert(
                2, "[progress.percentage]{task.percentage:>3.0f}%"
            )
            progress_bar_fields.insert(
                3,
                TimeRemainingColumn(),
            )
        self.progress_bar = Progress(
            *progress_bar_fields,
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self.task = self.progress_bar.add_task(
            self.abbreviate_description(description),
            total=total_frames,
            start=self.input_path is not None,
            process_fps=0,
        )

    def __enter__(self):
        """Return ``self`` so ``Video`` can be used in a ``with`` block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release video resources when leaving a ``with`` block."""
        self.close()
        return False

    def close(self):
        """Release video resources (VideoCapture, VideoWriter, GUI windows).

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._closed:
            return
        self._closed = True
        if self.output_video is not None:
            self.output_video.release()
            print(
                f"[white]Output video file saved to: {self.get_output_file_path()}[/white]"
            )
        self.video_capture.release()
        with contextlib.suppress(cv2.error):
            cv2.destroyAllWindows()

    # This is a generator, note the yield keyword below.
    def __iter__(self):
        """Yield successive OpenCV frames from the underlying capture.

        The progress bar is updated for every frame. When the
        generator is exhausted (or the consumer loop exits for any
        reason), :meth:`close` is invoked to release resources.

        Raises
        ------
        RuntimeError
            If the ``Video`` has already been closed.

        """
        if self._closed:
            raise RuntimeError("Cannot iterate over a closed Video")
        try:
            with self.progress_bar as progress_bar:
                start = time.monotonic()
                next_refresh_at = start

                # Iterate over video
                while True:
                    ret, frame = self.video_capture.read()
                    if ret is False or frame is None:
                        break
                    self.frame_counter += 1
                    now = time.monotonic()
                    elapsed = now - start
                    process_fps = self.frame_counter / elapsed if elapsed > 0 else 0.0
                    refresh_now = now >= next_refresh_at
                    progress_bar.update(
                        self.task,
                        advance=1,
                        refresh=refresh_now,
                        process_fps=process_fps,
                    )
                    if refresh_now:
                        next_refresh_at = now + _PROGRESS_REFRESH_INTERVAL_S
                    yield frame
                # Flush any throttled update so the bar ends at 100%.
                progress_bar.refresh()
        finally:
            self.close()

    def _fail(self, msg: str):
        """Raise a ``RuntimeError`` with ``msg``."""
        raise RuntimeError(msg)

    def write(self, frame: np.ndarray) -> int:
        """Write one frame to the output video.

        Lazily initializes the underlying ``cv2.VideoWriter`` on the
        first call, using ``frame``'s shape to determine the output
        resolution.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to write to file.

        Returns
        -------
        int
            The key code from ``cv2.waitKey(1)``.

        """
        if self.output_video is None:
            # The user may need to access the output file path on their code
            output_file_path = self.get_output_file_path()
            fourcc = _get_fourcc(self.get_codec_fourcc(output_file_path))
            # Set on first frame write in case the user resizes the frame in some way
            output_size = (
                frame.shape[1],
                frame.shape[0],
            )  # OpenCV format is (width, height)
            self.output_video = cv2.VideoWriter(
                output_file_path,
                fourcc,
                self.output_fps,
                output_size,
            )

        self.output_video.write(frame)
        return cv2.waitKey(1)

    def show(self, frame: np.ndarray, downsample_ratio: float = 1.0) -> int:
        """Display ``frame`` in a GUI window.

        Typically called inside a video-inference loop to preview the
        output.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to be displayed.
        downsample_ratio : float, optional
            Factor by which to downsample the frame before displaying.
            Useful when streaming the GUI video display over a slow
            connection (e.g. X11 forwarding over SSH).

        Returns
        -------
        int
            The key code from ``cv2.waitKey(1)``.

        """
        # Resize to lower resolution for faster streaming over slow connections
        if downsample_ratio != 1.0:
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1] / downsample_ratio),
                    int(frame.shape[0] / downsample_ratio),
                ),
            )
        cv2.imshow("Output", frame)
        return cv2.waitKey(1)

    def get_output_file_path(self) -> str:
        """Return the resolved output-file path for written frames.

        When ``output_path`` is a directory, Norfair auto-generates a
        filename based on the input source. This method returns that
        resolved path so callers can locate the rendered video.

        Returns
        -------
        str
            The absolute (or relative) path to the output file.

        """
        if not os.path.isdir(self.output_path):
            return self.output_path

        if self.input_path is not None:
            file_name = os.path.splitext(os.path.basename(self.input_path))[0]
        else:
            file_name = f"camera_{self.camera}"
        file_name = f"{file_name}_out.{self.output_extension}"

        return os.path.join(self.output_path, file_name)

    def get_codec_fourcc(self, filename: str) -> str:
        """Resolve the fourcc codec name to use when writing ``filename``.

        Returns :attr:`output_fourcc` when set; otherwise derives a
        default from ``filename``'s extension (``"XVID"`` for
        ``.avi``, ``"mp4v"`` for ``.mp4``).

        Parameters
        ----------
        filename : str
            Destination file whose extension is used for codec
            auto-selection.

        Returns
        -------
        str
            Name of the fourcc codec to feed to
            :class:`cv2.VideoWriter`.

        Raises
        ------
        RuntimeError
            If no codec can be resolved from ``filename``.

        """
        if self.output_fourcc is not None:
            return self.output_fourcc

        # Default codecs for each extension
        extension = os.path.splitext(filename)[1].lstrip(".").lower()
        if extension == "avi":
            return "XVID"
        elif extension == "mp4":
            return "mp4v"  # When available, "avc1" is better
        else:
            self._fail(
                f"[bold red]Could not determine video codec for the provided output filename[/bold red]: "
                f"[yellow]{filename}[/yellow]\n"
                f"Please use '.mp4', '.avi', or provide a custom OpenCV fourcc codec name."
            )
            # _fail raises RuntimeError, so this line is unreachable
            raise RuntimeError("Unreachable")

    def abbreviate_description(self, description: str) -> str:
        """Shorten ``description`` so the progress bar fits small terminals.

        Parameters
        ----------
        description : str
            The full description string to abbreviate.

        Returns
        -------
        str
            The original ``description`` if it fits, otherwise a
            truncated version with ``" ... "`` in the middle.
        """
        terminal_columns, _ = get_terminal_size()
        space_for_description = (
            int(terminal_columns) - 25
        )  # Leave 25 space for progressbar
        if len(description) < space_for_description:
            return description
        else:
            return f"{description[: space_for_description // 2 - 3]} ... {description[-space_for_description // 2 + 3 :]}"


class VideoFromFrames:
    """Iterate over an MOT-style dataset that stores frames as images.

    Reads ``seqinfo.ini`` to determine frame count, resolution and
    file-naming convention, then yields successive frames via the
    iterator protocol. Optionally encodes the frames into an ``.mp4``
    file alongside the iteration via :meth:`update`.

    Parameters
    ----------
    input_path : str
        Path to the MOT sequence directory (containing
        ``seqinfo.ini``).
    save_path : str, optional
        Directory where the rendered ``.mp4`` is written. A nested
        ``videos/`` folder is created if ``make_video`` is ``True``.
    information_file : metrics.InformationFile, optional
        Pre-parsed ``seqinfo.ini`` wrapper. Loaded from ``input_path``
        when not provided.
    make_video : bool, optional
        If ``True`` (the default), open a ``cv2.VideoWriter`` into
        which :meth:`update` will write frames.

    """

    def __init__(
        self, input_path, save_path=".", information_file=None, make_video=True
    ):
        """Parse ``seqinfo.ini`` and optionally open the output writer."""
        if information_file is None:
            information_file = metrics.InformationFile(
                file_path=os.path.join(input_path, "seqinfo.ini")
            )
        if make_video:
            file_name = os.path.split(input_path)[1]

            # Search framerate on seqinfo.ini
            fps_val = information_file.search(variable_name="frameRate")
            fps = float(fps_val) if isinstance(fps_val, (int, str)) else fps_val

            # Search resolution in seqinfo.ini
            h_res = information_file.search(variable_name="imWidth")
            v_res = information_file.search(variable_name="imHeight")
            horizontal_resolution = int(h_res) if not isinstance(h_res, int) else h_res
            vertical_resolution = int(v_res) if not isinstance(v_res, int) else v_res
            image_size = (horizontal_resolution, vertical_resolution)

            videos_folder = os.path.join(save_path, "videos")
            if not os.path.exists(videos_folder):
                os.makedirs(videos_folder)

            video_path = os.path.join(videos_folder, file_name + ".mp4")
            fourcc = _get_fourcc("mp4v")

            self.file_name = file_name
            # Video file
            self.video = cv2.VideoWriter(video_path, fourcc, fps, image_size)

        length_val = information_file.search(variable_name="seqLength")
        self.length = int(length_val) if not isinstance(length_val, int) else length_val
        self.input_path = input_path
        self.frame_number = 1
        ext_val = information_file.search("imExt")
        self.image_extension = str(ext_val) if not isinstance(ext_val, str) else ext_val
        dir_val = information_file.search("imDir")
        self.image_directory = str(dir_val) if not isinstance(dir_val, str) else dir_val

    def __iter__(self):
        """Reset the frame counter and return ``self`` as an iterator."""
        self.frame_number = 1
        return self

    def __next__(self):
        """Return the next image frame from the sequence."""
        if self.frame_number <= self.length:
            frame_path = os.path.join(
                self.input_path,
                self.image_directory,
                str(self.frame_number).zfill(6) + self.image_extension,
            )
            self.frame_number += 1

            return cv2.imread(frame_path)
        raise StopIteration()

    def update(self, frame):
        """Append ``frame`` to the output video and release on the last frame.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to write to the output video.
        """
        self.video.write(frame)
        cv2.waitKey(1)

        if self.frame_number > self.length:
            cv2.destroyAllWindows()
            self.video.release()
