"""Camera motion estimation module.

Contains the abstract coordinate transformation interfaces, the built-in
translation and homography implementations, and the :class:`MotionEstimator`
that ties them to OpenCV's sparse optical flow.
"""

import copy
import logging
from abc import ABC, abstractmethod

import numpy as np

try:
    import cv2
except ImportError:
    from .utils import DummyOpenCVImport

    cv2 = DummyOpenCVImport()

logger = logging.getLogger(__name__)


#
# Abstract interfaces
#
class CoordinatesTransformation(ABC):
    """Abstract base class representing a coordinate transformation.

    Detection and tracked-object coordinates can be interpreted in two
    reference frames:

    - **Relative** — their position on the current frame, where ``(0, 0)``
      is the top-left corner.
    - **Absolute** — their position in a fixed space, where ``(0, 0)`` is
      the top-left corner of the first frame of the video.

    A ``CoordinatesTransformation`` can map coordinates from one reference
    to the other.
    """

    @abstractmethod
    def abs_to_rel(self, points: np.ndarray) -> np.ndarray:
        """Map absolute-frame points to the current relative frame.

        Parameters
        ----------
        points : np.ndarray
            Array of shape ``(n_points, dim_points)`` in absolute
            coordinates.

        Returns
        -------
        np.ndarray
            Transformed points with the same shape, in relative
            coordinates.
        """

    @abstractmethod
    def rel_to_abs(self, points: np.ndarray) -> np.ndarray:
        """Map relative-frame points to the absolute frame.

        Parameters
        ----------
        points : np.ndarray
            Array of shape ``(n_points, dim_points)`` in relative
            coordinates.

        Returns
        -------
        np.ndarray
            Transformed points with the same shape, in absolute
            coordinates.
        """


class TransformationGetter(ABC):
    """Abstract base class for objects that infer a ``CoordinatesTransformation``.

    Subclasses take two point clouds (previous and current frame
    features) and return a flag indicating whether the reference frame
    should be reset together with the inferred transformation.
    """

    @abstractmethod
    def __call__(
        self, curr_pts: np.ndarray, prev_pts: np.ndarray
    ) -> tuple[bool, CoordinatesTransformation | None]:
        """Return ``(update_reference, transformation)`` for the current pair."""


#
# Translation
#
class TranslationTransformation(CoordinatesTransformation):
    """Coordinate transformation using a simple 2D translation.

    Parameters
    ----------
    movement_vector : np.ndarray
        The vector representing the translation.

    """

    def __init__(self, movement_vector):
        """Store the ``movement_vector`` used for the translation."""
        self.movement_vector = movement_vector

    def abs_to_rel(self, points: np.ndarray):
        """Translate absolute points into the current relative frame.

        Parameters
        ----------
        points : np.ndarray
            Array of shape ``(n_points, dim_points)`` in absolute
            coordinates.

        Returns
        -------
        np.ndarray
            Translated points with the same shape, in relative
            coordinates.
        """
        return points + self.movement_vector

    def rel_to_abs(self, points: np.ndarray):
        """Translate relative points back into the absolute frame.

        Parameters
        ----------
        points : np.ndarray
            Array of shape ``(n_points, dim_points)`` in relative
            coordinates.

        Returns
        -------
        np.ndarray
            Translated points with the same shape, in absolute
            coordinates.
        """
        return points - self.movement_vector


class TranslationTransformationGetter(TransformationGetter):
    """Compute a :class:`TranslationTransformation` from a pair of point clouds.

    The camera movement is estimated as the mode of the optical flow
    between the previous reference frame and the current one.

    Comparing consecutive frames can yield differences too small to
    estimate the translation reliably, so the reference frame is kept
    fixed as we progress through the video. Once the transformation can
    no longer match enough points, the reference frame is reset.

    Parameters
    ----------
    bin_size : float
        Before calculating the mode, optical-flow vectors are bucketized
        into bins of this size.
    proportion_points_used_threshold : float
        Proportion of points that must be matched; if the ratio drops
        below this value, the reference frame is updated.

    """

    def __init__(
        self, bin_size: float = 0.2, proportion_points_used_threshold: float = 0.9
    ) -> None:
        """Store parameters and initialize the running flow accumulator."""
        self.bin_size = bin_size
        self.proportion_points_used_threshold = proportion_points_used_threshold
        self.data = None

    def __call__(
        self, curr_pts: np.ndarray, prev_pts: np.ndarray
    ) -> tuple[bool, TranslationTransformation]:
        """Return the translation that best matches the optical flow."""
        # get flow
        flow = curr_pts - prev_pts

        # get mode
        flow = np.around(flow / self.bin_size) * self.bin_size
        unique_flows, counts = np.unique(flow, axis=0, return_counts=True)

        max_index = counts.argmax()

        proportion_points_used = counts[max_index] / len(prev_pts)
        update_prvs = proportion_points_used < self.proportion_points_used_threshold

        flow_mode = unique_flows[max_index]

        # Accumulate against the previously stored mode so we report the total
        # translation since the first frame. On the very first call `self.data`
        # is still None and there is nothing to accumulate against — leave the
        # freshly computed mode untouched.
        if self.data is not None:
            flow_mode += self.data

        if update_prvs:
            self.data = flow_mode

        return update_prvs, TranslationTransformation(flow_mode)


#
# Homography
#
class HomographyTransformation(CoordinatesTransformation):
    """Coordinate transformation using a 3×3 homography matrix.

    Parameters
    ----------
    homography_matrix : np.ndarray
        The matrix representing the homography.

    """

    def __init__(self, homography_matrix: np.ndarray):
        """Store the homography and pre-compute its inverse."""
        self.homography_matrix = homography_matrix
        self.inverse_homography_matrix = np.linalg.inv(homography_matrix)

    def abs_to_rel(self, points: np.ndarray):
        """Apply the forward homography to map absolute points to relative.

        Parameters
        ----------
        points : np.ndarray
            Array of shape ``(n_points, dim_points)`` or ``(dim_points,)``
            in absolute coordinates.

        Returns
        -------
        np.ndarray
            Transformed points with the same shape, in relative
            coordinates.
        """
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, -1)
        ones = np.ones((len(points), 1))
        points_with_ones = np.hstack((points, ones))
        points_transformed = points_with_ones @ self.homography_matrix.T
        last_column = points_transformed[:, -1]
        last_column = np.where(last_column == 0, np.finfo(float).eps, last_column)
        points_transformed = points_transformed / last_column.reshape(-1, 1)
        result = points_transformed[:, :2]
        if single_point:
            return result.flatten()
        return result

    def rel_to_abs(self, points: np.ndarray):
        """Apply the inverse homography to map relative points to absolute.

        Parameters
        ----------
        points : np.ndarray
            Array of shape ``(n_points, dim_points)`` or ``(dim_points,)``
            in relative coordinates.

        Returns
        -------
        np.ndarray
            Transformed points with the same shape, in absolute
            coordinates.
        """
        single_point = points.ndim == 1
        if single_point:
            points = points.reshape(1, -1)
        ones = np.ones((len(points), 1))
        points_with_ones = np.hstack((points, ones))
        points_transformed = points_with_ones @ self.inverse_homography_matrix.T
        last_column = points_transformed[:, -1]
        last_column = np.where(last_column == 0, np.finfo(float).eps, last_column)
        points_transformed = points_transformed / last_column.reshape(-1, 1)
        result = points_transformed[:, :2]
        if single_point:
            return result.flatten()
        return result


class HomographyTransformationGetter(TransformationGetter):
    """Compute a :class:`HomographyTransformation` from a pair of point clouds.

    The camera movement is represented as a homography that maps the
    optical flow between the previous reference frame and the current one.

    Comparing consecutive frames can make differences too small to
    estimate the homography reliably, often collapsing to the identity.
    The reference frame is therefore kept fixed as we progress through
    the video; once the transformation can no longer match enough points,
    it is reset.

    Parameters
    ----------
    method : int, optional
        One of OpenCV's methods for finding homographies. Valid options
        are ``[0, cv2.RANSAC, cv2.LMEDS, cv2.RHO]``. Defaults to
        ``cv2.RANSAC``.
    ransac_reproj_threshold : int, optional
        Maximum allowed reprojection error to treat a point pair as an
        inlier. See the OpenCV docs linked below for details.
    max_iters : int, optional
        The maximum number of RANSAC iterations. See the OpenCV docs
        linked below for details.
    confidence : float, optional
        Confidence level, must be between 0 and 1. See the OpenCV docs
        linked below for details.
    proportion_points_used_threshold : float, optional
        Proportion of points that must be matched; if the ratio drops
        below this value, the reference frame is updated.

    See Also
    --------
    [`cv2.findHomography`](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780)

    """

    def __init__(
        self,
        method: int | None = None,
        ransac_reproj_threshold: int = 3,
        max_iters: int = 2000,
        confidence: float = 0.995,
        proportion_points_used_threshold: float = 0.9,
    ) -> None:
        """Store RANSAC parameters and initialize the running homography."""
        self.data = None
        if method is None:
            method = cv2.RANSAC
        self.method = method
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.max_iters = max_iters
        self.confidence = confidence
        self.proportion_points_used_threshold = proportion_points_used_threshold

    def __call__(
        self, curr_pts: np.ndarray, prev_pts: np.ndarray
    ) -> tuple[bool, HomographyTransformation | None]:
        """Return the homography that best matches the optical flow."""
        if not (
            isinstance(prev_pts, np.ndarray)
            and prev_pts.shape[0] >= 4
            and isinstance(curr_pts, np.ndarray)
            and curr_pts.shape[0] >= 4
        ):
            logger.warning(
                "The homography couldn't be computed in this frame "
                "due to low amount of points"
            )
            if isinstance(self.data, np.ndarray):
                return True, HomographyTransformation(self.data)
            else:
                return True, None

        homography_matrix, points_used = cv2.findHomography(
            prev_pts,
            curr_pts,
            method=self.method,
            ransacReprojThreshold=self.ransac_reproj_threshold,
            maxIters=self.max_iters,
            confidence=self.confidence,
        )

        if homography_matrix is None or points_used is None:
            logger.warning(
                "Homography estimation failed for this frame "
                "(degenerate/collinear/insufficient inlier points)"
            )
            if isinstance(self.data, np.ndarray):
                return True, HomographyTransformation(self.data)
            else:
                return True, None

        proportion_points_used = np.sum(points_used) / len(points_used)

        update_prvs = proportion_points_used < self.proportion_points_used_threshold

        if self.data is not None:
            homography_matrix = homography_matrix @ self.data

        if update_prvs:
            self.data = homography_matrix

        return bool(update_prvs), HomographyTransformation(homography_matrix)


#
# Motion estimation
#
def _calc_optical_flow(
    prev_img: np.ndarray,
    next_img: np.ndarray,
    prev_pts: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Typed wrapper around ``cv2.calcOpticalFlowPyrLK``.

    OpenCV's ``calcOpticalFlowPyrLK`` can auto-initialize ``nextPts`` when
    it is empty. We pass an empty array of the correct dtype/shape to
    satisfy static type checkers while preserving the auto-initialization
    behavior.
    """
    # Create an empty array for nextPts; OpenCV will auto-initialize it
    next_pts_init: np.ndarray = np.array([], dtype=np.float32).reshape(0, 1, 2)
    result = cv2.calcOpticalFlowPyrLK(prev_img, next_img, prev_pts, next_pts_init)
    if result is None or len(result) < 2:
        return None, None
    return result[0], result[1]


def _get_sparse_flow(
    gray_next: np.ndarray,
    gray_prvs: np.ndarray,
    prev_pts: np.ndarray | None = None,
    max_points: int = 300,
    min_distance: int = 15,
    block_size: int = 3,
    mask: np.ndarray | None = None,
    quality_level: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Track a set of corner features between two grayscale frames."""
    if prev_pts is None:
        # get points
        prev_pts_result = cv2.goodFeaturesToTrack(
            gray_prvs,
            maxCorners=max_points,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
            mask=mask,
        )
        if prev_pts_result is None:
            # Return empty arrays if no features found
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
        prev_pts = prev_pts_result

    # compute optical flow
    curr_pts_result, status = _calc_optical_flow(gray_prvs, gray_next, prev_pts)

    # filter valid points
    if curr_pts_result is None or status is None:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

    idx = np.where(status == 1)[0]
    prev_pts_filtered = prev_pts[idx].reshape((-1, 2))
    curr_pts_filtered = curr_pts_result[idx].reshape((-1, 2))
    return curr_pts_filtered, prev_pts_filtered


class MotionEstimator:
    """Camera motion estimator driven by sparse optical flow.

    Uses OpenCV optical flow on a set of strong corner features to
    estimate the motion of the camera from frame to frame and feeds the
    result through a :class:`TransformationGetter` to recover a
    :class:`CoordinatesTransformation`.

    Parameters
    ----------
    max_points : int, optional
        Maximum number of points sampled. More points make the estimation
        slower but more precise.
    min_distance : int, optional
        Minimum distance between sampled points.
    block_size : int, optional
        Size of the averaging block used when finding corners. See the
        OpenCV link below for details.
    transformations_getter : TransformationGetter, optional
        The transformation estimator used on the sampled points. Defaults
        to
        [`HomographyTransformationGetter`][norfair.camera_motion.HomographyTransformationGetter].
    draw_flow : bool, optional
        Draw the optical flow on the frame in place, for debugging.
    flow_color : tuple[int, int, int], optional
        BGR color for the flow drawing. Defaults to a dark blue.
    quality_level : float, optional
        Minimum accepted quality of the image corners.

    Examples
    --------
    >>> from norfair import Tracker, Video
    >>> from norfair.camera_motion import MotionEstimator
    >>> video = Video(input_path="video.mp4")
    >>> tracker = Tracker(...)
    >>> motion_estimator = MotionEstimator()
    >>> for frame in video:
    ...     detections = get_detections(frame)
    ...     coord_transformation = motion_estimator.update(frame)
    ...     tracked_objects = tracker.update(
    ...         detections, coord_transformations=coord_transformation
    ...     )

    See Also
    --------
    [`cv2.goodFeaturesToTrack`](https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541)

    """

    def __init__(
        self,
        max_points: int = 200,
        min_distance: int = 15,
        block_size: int = 3,
        transformations_getter: TransformationGetter | None = None,
        draw_flow: bool = False,
        flow_color: tuple[int, int, int] | None = None,
        quality_level: float = 0.01,
    ):
        """Initialize sampling parameters and the transformation getter."""
        self.max_points = max_points
        self.min_distance = min_distance
        self.block_size = block_size

        self.draw_flow = draw_flow
        if self.draw_flow and flow_color is None:
            flow_color = (0, 0, 100)
        self.flow_color = flow_color

        self.gray_prvs = None
        self.prev_pts = None
        if transformations_getter is None:
            transformations_getter = HomographyTransformationGetter()

        self.transformations_getter = transformations_getter
        self.transformations_getter_copy = copy.deepcopy(transformations_getter)

        self.prev_mask = None
        self.gray_next = None
        self.quality_level = quality_level

    def update(
        self, frame: np.ndarray, mask: np.ndarray | None = None
    ) -> CoordinatesTransformation | None:
        """Estimate the camera motion for one frame.

        Parameters
        ----------
        frame : np.ndarray
            The current video frame.
        mask : np.ndarray, optional
            Optional mask excluding regions from corner sampling. Must
            have shape ``(frame.shape[0], frame.shape[1])``, dtype
            ``np.uint8``, and contain values ``0`` (ignore) or ``255``
            (consider), as required by ``cv2.goodFeaturesToTrack``.

            In general, the estimation works best when many points come
            from the background, so this parameter is useful for masking
            out detections or tracked objects and forcing the estimator
            to ignore moving objects. It can also be used to mask static
            overlays like sport scoreboards or security-camera
            timestamps.

        Returns
        -------
        CoordinatesTransformation or None
            A coordinate transformation that can map coordinates on this
            frame to absolute coordinates and vice versa, or ``None`` if
            the transformation could not be recovered.

        """
        self.gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.gray_prvs is None:
            self.gray_prvs = self.gray_next
            self.prev_mask = mask

        curr_pts, prev_pts = None, None
        try:
            curr_pts, prev_pts = _get_sparse_flow(
                self.gray_next,
                self.gray_prvs,
                self.prev_pts,
                self.max_points,
                self.min_distance,
                self.block_size,
                self.prev_mask,
                quality_level=self.quality_level,
            )
            if self.draw_flow and self.flow_color is not None:
                for curr, prev in zip(curr_pts, prev_pts):
                    c = tuple(curr.astype(int).ravel())
                    p = tuple(prev.astype(int).ravel())
                    cv2.line(frame, c, p, self.flow_color, 2)
                    cv2.circle(frame, c, 3, self.flow_color, -1)
        except (cv2.error, ValueError, TypeError) as e:
            logger.warning(e)

        update_prvs, coord_transformations = True, None
        if curr_pts is not None and prev_pts is not None:
            try:
                update_prvs, coord_transformations = self.transformations_getter(
                    curr_pts, prev_pts
                )
            except (TypeError, ValueError, np.linalg.LinAlgError) as e:
                logger.warning(e)
                del self.transformations_getter
                self.transformations_getter = copy.deepcopy(
                    self.transformations_getter_copy
                )

        if update_prvs:
            self.gray_prvs = self.gray_next
            self.prev_pts = None
            self.prev_mask = mask
        else:
            self.prev_pts = prev_pts

        return coord_transformations
