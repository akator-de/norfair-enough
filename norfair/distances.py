"""Predefined distance functions and the :class:`Distance` base class."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, overload

import numpy as np
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from typing import TypeAlias

    from .tracker import Detection, TrackedObject

    Candidate: TypeAlias = "Detection | TrackedObject"

logger = logging.getLogger(__name__)


class Distance(ABC):
    """Abstract base class representing a tracker distance.

    Subclasses must implement :meth:`get_distances`, which returns a
    distance matrix between tracked objects and candidates (detections or
    other tracked objects, when ReID is in use).
    """

    @abstractmethod
    def get_distances(
        self,
        objects: Sequence["TrackedObject"],
        candidates: Sequence["Candidate"] | None,
    ) -> np.ndarray:
        """Return the distance matrix between ``objects`` and ``candidates``.

        Parameters
        ----------
        objects : Sequence[TrackedObject]
            Sequence of [TrackedObject][norfair.tracker.TrackedObject]
            instances currently being tracked.
        candidates : Sequence[Detection or TrackedObject], optional
            Candidates to be compared against the tracked ``objects``.
            Detections are used during the normal matching step; tracked
            objects are used during ReID.

        Returns
        -------
        np.ndarray
            A ``(n_candidates, n_objects)`` matrix of distances.

        """


class ScalarDistance(Distance):
    """Distance computed pointwise (one pair at a time).

    Parameters
    ----------
    distance_function : Callable
        Function used to compute the distance between a pair. It must
        accept two positional arguments — a ``Detection`` or
        ``TrackedObject`` and a ``TrackedObject`` — and return a ``float``.

    """

    @overload
    def __init__(
        self,
        distance_function: Callable[["Detection", "TrackedObject"], float],
    ): ...

    @overload
    def __init__(
        self,
        distance_function: Callable[["TrackedObject", "TrackedObject"], float],
    ): ...

    def __init__(
        self,
        distance_function: Callable[["Detection", "TrackedObject"], float]
        | Callable[["TrackedObject", "TrackedObject"], float],
    ):
        """Store the per-pair ``distance_function``.

        The two overloads (detection→object and object→object) are both
        valid at runtime — Python's duck typing handles the dispatch.
        """
        self.distance_function: Callable = distance_function

    def get_distances(
        self,
        objects: Sequence["TrackedObject"],
        candidates: Sequence["Candidate"] | None,
    ) -> np.ndarray:
        """Return a distance matrix by calling ``distance_function`` for every pair.

        Pairs with mismatched labels are skipped and their entries left at
        ``np.inf``.

        Parameters
        ----------
        objects : Sequence[TrackedObject]
            Tracked objects to compare against ``candidates``.
        candidates : Sequence[Detection or TrackedObject], optional
            Candidates. ``None`` or empty sequences return a matrix filled
            with ``np.inf``.

        Returns
        -------
        np.ndarray
            A ``(n_candidates, n_objects)`` matrix of distances.

        """
        if not objects or not candidates:
            # Handle None or empty cases
            num_candidates = len(candidates) if candidates is not None else 0
            distance_matrix = np.full(
                (num_candidates, len(objects)),
                fill_value=np.inf,
                dtype=np.float32,
            )
            return distance_matrix

        distance_matrix = np.full(
            (len(candidates), len(objects)),
            fill_value=np.inf,
            dtype=np.float32,
        )
        for c, candidate in enumerate(candidates):
            for o, obj in enumerate(objects):
                if candidate.label != obj.label:
                    if (candidate.label is None) or (obj.label is None):
                        logger.warning(
                            "Label mismatch between candidate and tracked "
                            "object: candidate.label=%r, object.label=%r. "
                            "Mixing labelled and unlabelled inputs prevents "
                            "these pairs from ever matching.",
                            candidate.label,
                            obj.label,
                        )
                    continue
                distance = self.distance_function(candidate, obj)
                distance_matrix[c, o] = distance
        return distance_matrix


class VectorizedDistance(Distance):
    """Distance computed in a single vectorized operation.

    Rather than iterating over every pair of candidate and tracked object,
    ``VectorizedDistance`` stacks their coordinates and hands the whole
    batch to ``distance_function`` in one call — much faster for large
    numbers of objects.

    Parameters
    ----------
    distance_function : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Distance function that accepts two 2D arrays ``(candidates,
        objects)`` and returns a ``(n_candidates, n_objects)`` distance
        matrix.

    """

    def __init__(
        self,
        distance_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        """Store the vectorized ``distance_function``."""
        self.distance_function = distance_function

    def get_distances(
        self,
        objects: Sequence["TrackedObject"],
        candidates: Sequence["Candidate"] | None,
    ) -> np.ndarray:
        """Return the distance matrix computed per label group.

        Objects and candidates are grouped by label; for each label the
        corresponding sub-block of the distance matrix is filled by
        ``distance_function`` called on the stacked coordinates. Entries
        across different labels remain ``np.inf``.

        Parameters
        ----------
        objects : Sequence[TrackedObject]
            Tracked objects to compare against ``candidates``.
        candidates : Sequence[Detection or TrackedObject], optional
            Candidates. ``None`` or empty sequences return a matrix filled
            with ``np.inf``.

        Returns
        -------
        np.ndarray
            A ``(n_candidates, n_objects)`` matrix of distances.

        """
        if not objects or not candidates:
            # Handle None or empty cases
            num_candidates = len(candidates) if candidates is not None else 0
            distance_matrix = np.full(
                (num_candidates, len(objects)),
                fill_value=np.inf,
                dtype=np.float32,
            )
            return distance_matrix

        distance_matrix = np.full(
            (len(candidates), len(objects)),
            fill_value=np.inf,
            dtype=np.float32,
        )

        from .tracker import Detection

        object_labels = np.array([o.label for o in objects]).astype(str)
        candidate_labels = np.array([c.label for c in candidates]).astype(str)

        # iterate over labels that are present both in objects and detections
        for label in np.intersect1d(
            np.unique(object_labels), np.unique(candidate_labels)
        ):
            # generate masks of the subset of object and detections for this label
            obj_mask = object_labels == label
            cand_mask = candidate_labels == label

            # Use the already-computed boolean masks instead of re-comparing labels
            stacked_objects = np.stack(
                [o.estimate.ravel() for o, m in zip(objects, obj_mask) if m]
            )
            stacked_candidates = np.stack(
                [
                    c.points.ravel() if isinstance(c, Detection) else c.estimate.ravel()
                    for c, m in zip(candidates, cand_mask)
                    if m
                ]
            )

            # calculate the pairwise distances between objects and candidates with this label
            # and assign the result to the correct positions inside distance_matrix
            distance_matrix[np.ix_(cand_mask, obj_mask)] = self._compute_distance(
                stacked_candidates, stacked_objects
            )

        return distance_matrix

    def _compute_distance(
        self, stacked_candidates: np.ndarray, stacked_objects: np.ndarray
    ) -> np.ndarray:
        """Compute the pairwise distance between stacked candidates and objects.

        Parameters
        ----------
        stacked_candidates : np.ndarray
            Stacked candidate coordinates.
        stacked_objects : np.ndarray
            Stacked object coordinates.

        Returns
        -------
        np.ndarray
            A ``(n_candidates, n_objects)`` matrix of distances.

        """
        return self.distance_function(stacked_candidates, stacked_objects)


class ScipyDistance(VectorizedDistance):
    """Vectorized distance backed by ``scipy.spatial.distance.cdist``.

    Uses [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    to calculate distances between two ``np.ndarray`` batches.

    Parameters
    ----------
    metric : str, optional
        Defines the specific Scipy metric to use to calculate the pairwise distances between
        new candidates and objects.
    **kwargs
        Additional keyword arguments forwarded to
        `scipy.spatial.distance.cdist`.

    See Also
    --------
    [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

    """

    def __init__(self, metric: str = "euclidean", **kwargs):
        """Configure the scipy metric and any extra ``cdist`` keyword arguments."""
        self.metric = metric
        super().__init__(distance_function=partial(cdist, metric=self.metric, **kwargs))


def frobenius(detection: "Detection", tracked_object: "TrackedObject") -> float:
    r"""Frobenius norm of the difference between detection points and tracked-object estimates.

    The Frobenius distance and norm are given by:

    $$
    d_f(a, b) = ||a - b||_F
    $$

    $$
    ||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}
    $$

    Parameters
    ----------
    detection : Detection
        A detection.
    tracked_object : TrackedObject
        A tracked object.

    Returns
    -------
    float
        The distance.

    See Also
    --------
    [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)

    """
    return float(np.linalg.norm(detection.points - tracked_object.estimate))


def mean_euclidean(detection: "Detection", tracked_object: "TrackedObject") -> float:
    r"""Average Euclidean distance between detection points and tracked-object estimates.

    $$
    d(a, b) = \frac{\sum_{i=0}^N ||a_i - b_i||_2}{N}
    $$

    Parameters
    ----------
    detection : Detection
        A detection.
    tracked_object : TrackedObject
        A tracked object.

    Returns
    -------
    float
        The distance.

    See Also
    --------
    [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)

    """
    return np.linalg.norm(detection.points - tracked_object.estimate, axis=1).mean()


def mean_manhattan(detection: "Detection", tracked_object: "TrackedObject") -> float:
    r"""Average Manhattan distance between detection points and tracked-object estimates.

    $$
    d(a, b) = \frac{\sum_{i=0}^N ||a_i - b_i||_1}{N}
    $$

    Where $||a||_1$ is the Manhattan norm.

    Parameters
    ----------
    detection : Detection
        A detection.
    tracked_object : TrackedObject
        A tracked object.

    Returns
    -------
    float
        The distance.

    See Also
    --------
    [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)

    """
    return np.linalg.norm(
        detection.points - tracked_object.estimate, ord=1, axis=1
    ).mean()


def _boxes_area(boxes: np.ndarray) -> np.ndarray:
    """Return the area of each bounding box in ``boxes``."""
    return (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])


def _validate_bboxes(bboxes: np.ndarray):
    """Validate that ``bboxes`` is a well-formed ``(N, 4)`` array of boxes."""
    if not (
        isinstance(bboxes, np.ndarray)
        and len(bboxes.shape) == 2
        and bboxes.shape[1] == 4
    ):
        raise ValueError(
            f"Bounding boxes must be defined as np.array with (N, 4) shape, {bboxes} given"
        )

    if not (all(bboxes[:, 0] < bboxes[:, 2]) and all(bboxes[:, 1] < bboxes[:, 3])):
        logger.warning(
            "Incorrect bounding boxes. Check that x_min < x_max and y_min < y_max."
        )


def iou(candidates: np.ndarray, objects: np.ndarray) -> np.ndarray:
    """Compute ``1 - IoU`` between two sets of bounding boxes.

    Both sets of boxes are expected to be in
    ``[x_min, y_min, x_max, y_max]`` format.

    Normal IoU is ``1`` when the boxes are identical and ``0`` when they
    don't overlap; to turn this into a distance the function returns
    ``1 - IoU``.

    Parameters
    ----------
    candidates : np.ndarray
        ``(N, 4)`` array of candidate bounding boxes.
    objects : np.ndarray
        ``(K, 4)`` array of object bounding boxes.

    Returns
    -------
    np.ndarray
        ``(N, K)`` array of ``1 - IoU`` values between candidates and
        objects.

    """
    _validate_bboxes(candidates)
    _validate_bboxes(objects)

    area_candidates = _boxes_area(candidates.T)
    area_objects = _boxes_area(objects.T)

    top_left = np.maximum(candidates[:, None, :2], objects[:, :2])
    bottom_right = np.minimum(candidates[:, None, 2:], objects[:, 2:])

    area_intersection = np.prod(
        np.clip(bottom_right - top_left, a_min=0, a_max=None), 2
    )
    return 1 - area_intersection / (
        area_candidates[:, None] + area_objects - area_intersection
    )


iou_opt = iou  # deprecated


_SCALAR_DISTANCE_FUNCTIONS = {
    "frobenius": frobenius,
    "mean_manhattan": mean_manhattan,
    "mean_euclidean": mean_euclidean,
}
_VECTORIZED_DISTANCE_FUNCTIONS = {
    "iou": iou,
    "iou_opt": iou,  # deprecated
}
_SCIPY_DISTANCE_FUNCTIONS = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulczynski1",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]
AVAILABLE_VECTORIZED_DISTANCES = (
    list(_VECTORIZED_DISTANCE_FUNCTIONS.keys()) + _SCIPY_DISTANCE_FUNCTIONS
)


def get_distance_by_name(name: str) -> Distance:
    """Return a predefined :class:`Distance` by name.

    Accepts the names of Norfair's built-in scalar and vectorized
    distances, as well as any metric supported by
    [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html).

    Parameters
    ----------
    name : str
        Name of the distance to look up.

    Returns
    -------
    Distance
        A distance object ready to be passed to ``Tracker``.

    Raises
    ------
    ValueError
        If ``name`` is not a known distance.

    """
    distance_function: Distance
    if name in _SCALAR_DISTANCE_FUNCTIONS:
        logger.warning(
            "You are using a scalar distance function. If you want to speed up the"
            " tracking process please consider using a vectorized distance function"
            f" such as {AVAILABLE_VECTORIZED_DISTANCES}."
        )
        distance_function = ScalarDistance(_SCALAR_DISTANCE_FUNCTIONS[name])
    elif name in _SCIPY_DISTANCE_FUNCTIONS:
        distance_function = ScipyDistance(name)
    elif name in _VECTORIZED_DISTANCE_FUNCTIONS:
        if name == "iou_opt":
            logger.warning("iou_opt is deprecated, use iou instead")
        distance_function = VectorizedDistance(_VECTORIZED_DISTANCE_FUNCTIONS[name])
    else:
        raise ValueError(
            f"Invalid distance '{name}', expecting one of"
            f" {list(_SCALAR_DISTANCE_FUNCTIONS.keys()) + AVAILABLE_VECTORIZED_DISTANCES}"
        )

    return distance_function


def create_keypoints_voting_distance(
    keypoint_distance_threshold: float, detection_threshold: float
) -> Callable[["Detection", "TrackedObject"], float]:
    """Build a keypoint-voting scalar distance bound to the given thresholds.

    The returned distance counts how many points in a detection match the
    points in a tracked object. A point counts as a match when the
    distance between it and its peer is below
    ``keypoint_distance_threshold`` and both detection and tracked-object
    scores exceed ``detection_threshold``. The ``i``-th point in a
    detection can only match the ``i``-th point in a tracked object.

    The distance is ``1`` when nothing matches and tends towards ``0`` as
    more points match.

    Parameters
    ----------
    keypoint_distance_threshold : float
        Points closer than this threshold count as a match.
    detection_threshold : float
        Points with score at or below this threshold are ignored.

    Returns
    -------
    Callable
        A scalar distance function that can be passed to ``Tracker``.

    """

    def keypoints_voting_distance(
        detection: "Detection", tracked_object: "TrackedObject"
    ) -> float:
        if detection.scores is None or tracked_object.last_detection.scores is None:
            return 1.0
        distances = np.linalg.norm(detection.points - tracked_object.estimate, axis=1)
        match_num = np.count_nonzero(
            (distances < keypoint_distance_threshold)
            * (detection.scores > detection_threshold)
            * (tracked_object.last_detection.scores > detection_threshold)
        )
        return 1 / (1 + match_num)

    return keypoints_voting_distance


def create_normalized_mean_euclidean_distance(
    height: int, width: int
) -> Callable[["Detection", "TrackedObject"], float]:
    """Build a normalized mean Euclidean distance bound to the image size.

    The returned distance is normalized so it lies in ``[0, 1]``, where
    ``1`` corresponds to opposite corners of the image.

    Parameters
    ----------
    height : int
        Height of the image.
    width : int
        Width of the image.

    Returns
    -------
    Callable
        A scalar distance function that can be passed to ``Tracker``.

    """

    def normalized__mean_euclidean_distance(
        detection: "Detection", tracked_object: "TrackedObject"
    ) -> float:
        """Normalized mean Euclidean distance between detection and tracked object."""
        # calculate distances and normalized it by width and height
        difference = (detection.points - tracked_object.estimate).astype(float)
        difference[:, 0] /= width
        difference[:, 1] /= height

        # calculate eucledean distance and average
        return np.linalg.norm(difference, axis=1).mean()

    return normalized__mean_euclidean_distance


__all__ = [
    "frobenius",
    "mean_manhattan",
    "mean_euclidean",
    "iou",
    "iou_opt",
    "get_distance_by_name",
    "create_keypoints_voting_distance",
    "create_normalized_mean_euclidean_distance",
]
