"""Video stabilization drawer built on camera-motion estimates."""

import numpy as np

from norfair.camera_motion import TranslationTransformation
from norfair.utils import warn_once


class FixedCamera:
    """Stabilize the video by compensating for estimated camera motion.

    The drawer renders on a larger canvas and shifts the original
    frame in the opposite direction of the camera motion, so stationary
    objects in the world stay pinned in the output. Useful for
    debugging or showcasing camera-motion estimation.

    !!! Warning
        Only supports
        [`TranslationTransformation`][norfair.camera_motion.TranslationTransformation].
        Passing a
        [`HomographyTransformation`][norfair.camera_motion.HomographyTransformation]
        yields undefined behavior.

    !!! Warning
        If combined with other drawers, always apply ``FixedCamera``
        last. Drawing on the scaled-up frame produced by this class
        will not give the expected result.

    !!! Note
        Sometimes the camera moves so far from the starting point that
        the shifted frame no longer fits inside the scaled-up canvas.
        In that case, a warning is logged and the frame is cropped.

    Parameters
    ----------
    scale : float, optional
        The output resolution is ``scale * (H, W)`` where ``H, W`` is
        the resolution of the input frame. Increase this when the
        camera moves a lot.
    attenuation : float, optional
        Controls how quickly older content fades toward black.

    Examples
    --------
    Stabilize a video using ``FixedCamera`` alongside a tracker::

        >>> tracker = Tracker("frobenius", 100)
        >>> motion_estimator = MotionEstimator()
        >>> fixed_camera = FixedCamera()
        >>> with Video(input_path="video.mp4") as video:
        ...     for frame in video:
        ...         coord_transformations = motion_estimator.update(frame)
        ...         detections = get_detections(frame)
        ...         tracked_objects = tracker.update(
        ...             detections, coord_transformations=coord_transformations
        ...         )
        ...         # apply fixed_camera last
        ...         draw_points(frame, tracked_objects)
        ...         bigger_frame = fixed_camera.adjust_frame(
        ...             frame, coord_transformations
        ...         )
        ...         video.write(bigger_frame)

    """

    def __init__(self, scale: float = 2, attenuation: float = 0.05):
        """Initialize the background canvas parameters."""
        self.scale = scale
        self._background: np.ndarray | None = None
        self._attenuation_factor = 1 - attenuation

    def adjust_frame(
        self, frame: np.ndarray, coord_transformation: TranslationTransformation
    ) -> np.ndarray:
        """Render the next frame onto the stabilized background canvas.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame for this time step.
        coord_transformation : TranslationTransformation
            The coordinate transformation as returned by the
            [`MotionEstimator`][norfair.camera_motion.MotionEstimator]
            for this frame.

        Returns
        -------
        np.ndarray
            The scaled-up background canvas with ``frame`` drawn onto
            it at the position that compensates for the estimated
            camera motion.

        """

        # initialize background if necessary
        background: np.ndarray
        if self._background is None:
            original_size = (
                frame.shape[1],
                frame.shape[0],
            )  # OpenCV format is (width, height)

            scaled_size = tuple(
                (np.array(original_size) * np.array(self.scale)).round().astype(int)
            )
            background = np.zeros(
                [scaled_size[1], scaled_size[0], frame.shape[-1]],
                frame.dtype,
            )
        else:
            background = (self._background * self._attenuation_factor).astype(
                frame.dtype
            )

        # top_left is the anchor coordinate from where we start drawing the fame on top of the background
        # aim to draw it in the center of the background but transformations will move this point
        top_left = np.array(background.shape[:2]) // 2 - np.array(frame.shape[:2]) // 2
        top_left = (
            coord_transformation.rel_to_abs(top_left[::-1]).round().astype(int)[::-1]
        )
        # box of the background that will be updated and the limits of it
        background_y0, background_y1 = (top_left[0], top_left[0] + frame.shape[0])
        background_x0, background_x1 = (top_left[1], top_left[1] + frame.shape[1])
        background_size_y, background_size_x = background.shape[:2]

        # define box of the frame that will be used
        # if the scale is not enough to support the movement, warn the user but keep drawing
        # cropping the frame so that the operation doesn't fail
        frame_y0, frame_y1, frame_x0, frame_x1 = (0, frame.shape[0], 0, frame.shape[1])
        if (
            background_y0 < 0
            or background_x0 < 0
            or background_y1 > background_size_y
            or background_x1 > background_size_x
        ):
            warn_once(
                "moving_camera_scale is not enough to cover the range of camera movement, frame will be cropped"
            )
            # crop left or top of the frame if necessary
            frame_y0 = max(-background_y0, 0)
            frame_x0 = max(-background_x0, 0)
            # crop right or bottom of the frame if necessary
            frame_y1 = max(
                min(background_size_y - background_y0, background_y1 - background_y0), 0
            )
            frame_x1 = max(
                min(background_size_x - background_x0, background_x1 - background_x0), 0
            )
            # handle cases where the limits of the background become negative which numpy will interpret incorrectly
            background_y0 = max(background_y0, 0)
            background_x0 = max(background_x0, 0)
            background_y1 = max(background_y1, 0)
            background_x1 = max(background_x1, 0)
        background[background_y0:background_y1, background_x0:background_x1, :] = frame[
            frame_y0:frame_y1, frame_x0:frame_x1, :
        ]
        self._background = background
        return background
