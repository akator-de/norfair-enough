"""Tracking primitives: ``Tracker``, ``TrackedObject`` and ``Detection``."""

import copy
import logging
import threading
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from norfair.camera_motion import CoordinatesTransformation

from .distances import (
    AVAILABLE_VECTORIZED_DISTANCES,
    Distance,
    ScalarDistance,
    get_distance_by_name,
)
from .filter import FilterFactory, OptimizedKalmanFilterFactory
from .utils import validate_points

logger = logging.getLogger(__name__)


class Tracker:
    """Tracks detections produced by a detector across frames.

    Parameters
    ----------
    distance_function : str or Callable[[Detection, TrackedObject], float]
        Function used by the tracker to determine the distance between newly
        detected objects and the objects that are currently being tracked.
        This function should take 2 input arguments, the first being a
        [Detection][norfair.tracker.Detection], and the second a
        [TrackedObject][norfair.tracker.TrackedObject]. It has to return a
        `float` with the distance it calculates.

        Some common distances are implemented in [distances][]; as a shortcut
        the tracker accepts the name of one of these
        [predefined distances][norfair.distances.get_distance_by_name]. Scipy's
        predefined distances are also accepted — pass a `str` with one of the
        available metrics in
        [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html).
    distance_threshold : float
        Maximum distance that can constitute a match. Detections and tracked
        objects whose distance is above this threshold won't be matched by
        the tracker.
    hit_counter_max : int, optional
        Each tracked object keeps an internal hit counter which tracks how
        often it gets matched to a detection; each time it gets a match the
        counter goes up, and each time it doesn't the counter goes down.

        If the counter goes below 0 the object gets destroyed. This argument
        defines how large this inertia can grow, and therefore how long an
        object can live without getting matched to any detections before it
        is displaced as a dead object. If no ReID distance function is
        provided the object will then be destroyed.
    initialization_delay : int, optional
        Determines how large the object's hit counter must be in order to be
        considered initialized and be returned to the user as a real object.
        It must be smaller than ``hit_counter_max`` or otherwise the object
        would never be initialized.

        If set to 0, objects will be returned to the user as soon as they are
        detected for the first time, which can be problematic because it can
        result in objects appearing and immediately disappearing.

        Defaults to ``hit_counter_max // 2``.
    pointwise_hit_counter_max : int, optional
        Each tracked object keeps track of how often the points it is
        tracking have been getting matched. Points that are getting matched
        (``pointwise_hit_counter > 0``) are said to be *live*, while points
        that aren't (``pointwise_hit_counter == 0``) are said to be stale.

        This is used to determine things like which individual points in a
        tracked object get drawn by
        [`draw_points`][norfair.drawing.draw_points] and which don't.
        This argument defines how large the inertia for each point of a
        tracker can grow.
    detection_threshold : float, optional
        Threshold at which point scores in a detection must be above to be
        considered by the tracker. Any point whose score is at or below this
        value is ignored.
    filter_factory : FilterFactory, optional
        Selects which filter the
        [`TrackedObject`][norfair.tracker.TrackedObject] instances created by
        this tracker will use. Defaults to
        [`OptimizedKalmanFilterFactory()`][norfair.filter.OptimizedKalmanFilterFactory].
    past_detections_length : int, optional
        How many past detections to save for each tracked object. Norfair
        tries to distribute these past detections uniformly through the
        object's lifetime so they're more representative. Very useful if you
        want to add metric learning to your model, because you can associate
        an embedding to each detection and access them in your distance
        function.
    reid_distance_function : Callable[[TrackedObject, TrackedObject], float], optional
        Function used by the tracker to determine the ReID distance between
        newly detected trackers and unmatched trackers.

        This function should take 2 input arguments, the first being a
        tracked object in the initialization phase, and the second being a
        tracked object that has been unmatched. Both are
        [`TrackedObject`][norfair.tracker.TrackedObject] instances. It must
        return a `float` with the distance it calculates.
    reid_distance_threshold : float, optional
        Maximum ReID distance that can constitute a match.

        Tracked objects whose distance is above this threshold won't be
        merged. If they are merged, the oldest tracked object is maintained
        with the position of the new tracked object.
    reid_hit_counter_max : int, optional
        Each tracked object keeps an internal ReID hit counter which tracks
        how often it is getting recognized by another tracker; each time it
        gets a match this counter goes up, and each time it doesn't it goes
        down. If it goes below 0 the object is destroyed. When set, this
        defines how long an object can live without being matched to any
        detection before it is destroyed.

    Notes
    -----
    Individual ``Tracker`` instances are **not** thread-safe; do not call
    ``update`` on the same instance from multiple threads. However, using
    separate ``Tracker`` instances in different threads is safe — the shared
    ``global_id`` counter is protected by a lock.

    """

    def __init__(
        self,
        distance_function: str | Callable[["Detection", "TrackedObject"], float],
        distance_threshold: float,
        hit_counter_max: int = 15,
        initialization_delay: int | None = None,
        pointwise_hit_counter_max: int = 4,
        detection_threshold: float = 0,
        filter_factory: FilterFactory | None = None,
        past_detections_length: int = 4,
        reid_distance_function: Callable[["TrackedObject", "TrackedObject"], float]
        | None = None,
        reid_distance_threshold: float = 0,
        reid_hit_counter_max: int | None = None,
    ):
        self.tracked_objects: list[TrackedObject] = []

        if filter_factory is None:
            filter_factory = OptimizedKalmanFilterFactory()

        # Convert distance_function to Distance object
        distance_obj: Distance
        if isinstance(distance_function, str):
            distance_obj = get_distance_by_name(distance_function)
        elif callable(distance_function):
            logger.warning(
                "You are using a scalar distance function. If you want to speed up the"
                " tracking process please consider using a vectorized distance"
                f" function such as {AVAILABLE_VECTORIZED_DISTANCES}."
            )
            distance_obj = ScalarDistance(distance_function)
        else:
            raise ValueError(
                "Argument `distance_function` should be a string or function but is"
                f" {type(distance_function)} instead."
            )
        self.distance_function = distance_obj

        self.hit_counter_max = hit_counter_max
        self.reid_hit_counter_max = reid_hit_counter_max
        self.pointwise_hit_counter_max = pointwise_hit_counter_max
        self.filter_factory = filter_factory
        if past_detections_length >= 0:
            self.past_detections_length = past_detections_length
        else:
            raise ValueError(
                f"Argument `past_detections_length` is {past_detections_length} and should be 0 or larger."
            )

        if initialization_delay is None:
            self.initialization_delay = int(self.hit_counter_max / 2)
        elif initialization_delay < 0 or initialization_delay >= self.hit_counter_max:
            raise ValueError(
                f"Argument 'initialization_delay' for 'Tracker' class should be an int between 0 and (hit_counter_max = {hit_counter_max}). The selected value is {initialization_delay}.\n"
            )
        else:
            self.initialization_delay = initialization_delay

        self.distance_threshold = distance_threshold
        self.detection_threshold = detection_threshold
        self.reid_distance_function: ScalarDistance | None
        if reid_distance_function is not None:
            self.reid_distance_function = ScalarDistance(reid_distance_function)
        else:
            self.reid_distance_function = None
        self.reid_distance_threshold = reid_distance_threshold
        self._obj_factory = _TrackedObjectFactory()

    def update(
        self,
        detections: list["Detection"] | None = None,
        period: int = 1,
        coord_transformations: CoordinatesTransformation | None = None,
    ) -> list["TrackedObject"]:
        """Process detections found in the current frame.

        Detections can be matched to previous tracked objects, or new
        tracked objects will be created according to the configuration of
        this ``Tracker``. The currently alive and initialized tracked
        objects are returned.

        Parameters
        ----------
        detections : list[Detection], optional
            The [`Detection`][norfair.tracker.Detection] instances found in
            the current frame being processed.

            If no detections were found in the current frame, or the user is
            purposely skipping frames to improve processing time, this
            argument should be set to ``None`` or omitted — the update
            function still needs to be called to advance the state of the
            Kalman filters inside the tracker.
        period : int, optional
            Many users choose not to run their detector on every frame in
            order to process video faster. This parameter sets how many
            frames pass between detector invocations, so the tracker is
            aware and can handle the situation properly.

            It can be reset on each frame processed, which is useful if you
            are dynamically changing how many frames the detector skips in a
            real-time video stream.
        coord_transformations : CoordinatesTransformation, optional
            The coordinate transformation calculated by the
            [`MotionEstimator`][norfair.camera_motion.MotionEstimator].

        Returns
        -------
        list[TrackedObject]
            The list of active tracked objects.

        Notes
        -----
        When *coord_transformations* is provided this method **mutates**
        ``Detection.absolute_points`` on the caller's objects (overwritten
        with absolute coordinates).  All other ``Detection`` attributes
        (including ``age``) are left untouched; the tracker stores shallow
        copies of detections internally.

        """
        if coord_transformations is not None and detections is not None:
            for det in detections:
                det.update_coordinate_transformation(coord_transformations)

        # Remove stale trackers and make candidate object real if the hit counter is positive
        alive_objects = []
        dead_objects = []
        if self.reid_hit_counter_max is None:
            self.tracked_objects = [
                o for o in self.tracked_objects if o.hit_counter_is_positive
            ]
            alive_objects = self.tracked_objects
        else:
            tracked_objects = []
            for o in self.tracked_objects:
                if o.reid_hit_counter_is_positive:
                    tracked_objects.append(o)
                    if o.hit_counter_is_positive:
                        alive_objects.append(o)
                    else:
                        dead_objects.append(o)
            self.tracked_objects = tracked_objects

        # Update tracker
        for obj in self.tracked_objects:
            obj.tracker_step()
            obj.update_coordinate_transformation(coord_transformations)

        # Update initialized tracked objects with detections
        (
            unmatched_detections,
            _,
            unmatched_init_trackers,
        ) = self._update_objects_in_place(
            self.distance_function,
            self.distance_threshold,
            [o for o in alive_objects if not o.is_initializing],
            detections,
            period,
        )

        # Update not yet initialized tracked objects with yet unmatched detections
        (
            unmatched_detections,
            matched_not_init_trackers,
            _,
        ) = self._update_objects_in_place(
            self.distance_function,
            self.distance_threshold,
            [o for o in alive_objects if o.is_initializing],
            unmatched_detections,
            period,
        )

        if self.reid_distance_function is not None:
            # Match unmatched initialized tracked objects with not yet initialized tracked objects
            _, _, _ = self._update_objects_in_place(
                self.reid_distance_function,
                self.reid_distance_threshold,
                unmatched_init_trackers + dead_objects,
                matched_not_init_trackers,
                period,
            )

        # Create new tracked objects from remaining unmatched detections
        for detection in unmatched_detections:
            if not isinstance(detection, Detection):
                continue
            self.tracked_objects.append(
                self._obj_factory.create(
                    initial_detection=detection,
                    hit_counter_max=self.hit_counter_max,
                    initialization_delay=self.initialization_delay,
                    pointwise_hit_counter_max=self.pointwise_hit_counter_max,
                    detection_threshold=self.detection_threshold,
                    period=period,
                    filter_factory=self.filter_factory,
                    past_detections_length=self.past_detections_length,
                    reid_hit_counter_max=self.reid_hit_counter_max,
                    coord_transformations=coord_transformations,
                )
            )

        return self.get_active_objects()

    @property
    def current_object_count(self) -> int:
        """Number of currently active ``TrackedObject`` instances."""
        return len(self.get_active_objects())

    @property
    def total_object_count(self) -> int:
        """Total number of ``TrackedObject`` instances ever initialized by this tracker."""
        return self._obj_factory.count

    def get_active_objects(self) -> list["TrackedObject"]:
        """Return the list of currently active tracked objects.

        An object is active if it has finished initializing and its hit
        counter is still positive.

        Returns
        -------
        list[TrackedObject]
            The active tracked objects.

        """
        return [
            o
            for o in self.tracked_objects
            if not o.is_initializing and o.hit_counter_is_positive
        ]

    def _update_objects_in_place(
        self,
        distance_function: Distance,
        distance_threshold: float,
        objects: Sequence["TrackedObject"],
        candidates: Sequence["Detection | TrackedObject"] | None,
        period: int,
    ) -> tuple[
        list["Detection | TrackedObject"],
        list["TrackedObject"],
        list["TrackedObject"],
    ]:
        if candidates is not None and len(candidates) > 0:
            distance_matrix = distance_function.get_distances(objects, candidates)
            if np.isnan(distance_matrix).any():
                raise ValueError(
                    "\nReceived nan values from distance function, please check your distance function for errors!"
                )

            # Used just for debugging distance function
            if distance_matrix.size > 0:
                for i, minimum in enumerate(distance_matrix.min(axis=0)):
                    objects[i].current_min_distance = (
                        minimum if minimum < distance_threshold else None
                    )

            matched_cand_indices, matched_obj_indices = self.match_dets_and_objs(
                distance_matrix, distance_threshold
            )
            if len(matched_cand_indices) > 0:
                unmatched_candidates = [
                    d for i, d in enumerate(candidates) if i not in matched_cand_indices
                ]
                unmatched_objects = [
                    d for i, d in enumerate(objects) if i not in matched_obj_indices
                ]
                matched_objects = []
                candidates_to_remove: set[int] = set()

                # Handle matched people/detections
                for match_cand_idx, match_obj_idx in zip(
                    matched_cand_indices, matched_obj_indices
                ):
                    match_distance = distance_matrix[match_cand_idx, match_obj_idx]
                    matched_candidate = candidates[match_cand_idx]
                    matched_object = objects[match_obj_idx]
                    if match_distance < distance_threshold:
                        if isinstance(matched_candidate, Detection):
                            matched_object.hit(matched_candidate, period=period)
                            matched_object.last_distance = match_distance
                            matched_objects.append(matched_object)
                        elif isinstance(matched_candidate, TrackedObject):
                            # Merge new TrackedObject with the old one
                            matched_object.merge(matched_candidate)
                            # Collect for batch removal instead of O(n) per-call list.remove()
                            candidates_to_remove.add(id(matched_candidate))
                    else:
                        unmatched_candidates.append(matched_candidate)
                        unmatched_objects.append(matched_object)

                # Batch-remove merged TrackedObject candidates in a single pass (O(n))
                if candidates_to_remove:
                    self.tracked_objects = [
                        o
                        for o in self.tracked_objects
                        if id(o) not in candidates_to_remove
                    ]
            else:
                unmatched_candidates = list(candidates)
                matched_objects = []
                unmatched_objects = list(objects)
        else:
            unmatched_candidates = []
            matched_objects = []
            unmatched_objects = list(objects)

        return unmatched_candidates, matched_objects, unmatched_objects

    def match_dets_and_objs(
        self, distance_matrix: NDArray[np.float64], distance_threshold
    ):
        """Match detections with tracked objects from a distance matrix.

        Instead of minimizing the global distance, this greedy strategy
        starts with the global minimum entry and matches the det-obj
        corresponding to that distance, then takes the second minimum, and
        so on until ``distance_threshold`` is reached.

        This avoids pathological cases where minimizing the global distance
        forces matches that shouldn't happen just to bring the overall sum
        down.

        The distances are pre-sorted with ``np.argsort`` so the scan is
        O(n*m*log(n*m) + n*m) instead of the previous O(min(n,m)*n*m)
        repeated-argmin approach.

        Parameters
        ----------
        distance_matrix : np.ndarray
            A matrix of shape ``(n_detections, n_objects)``.
        distance_threshold : float
            Entries greater than or equal to this value are not matched.

        Returns
        -------
        tuple[list[int], list[int]]
            Matched detection and object indices, in matching order.

        """
        if distance_matrix.size > 0:
            flat = distance_matrix.ravel()
            order = np.argsort(flat, kind="stable")
            ncols = distance_matrix.shape[1]

            det_idxs: list[int] = []
            obj_idxs: list[int] = []
            used_rows: set[int] = set()
            used_cols: set[int] = set()

            for idx in order:
                if flat[idx] >= distance_threshold:
                    break
                r, c = divmod(int(idx), ncols)
                if r not in used_rows and c not in used_cols:
                    det_idxs.append(r)
                    obj_idxs.append(c)
                    used_rows.add(r)
                    used_cols.add(c)

            return det_idxs, obj_idxs
        else:
            return [], []


class _TrackedObjectFactory:
    global_count = 0
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.count = 0
        self.initializing_count = 0

    def create(
        self,
        initial_detection: "Detection",
        hit_counter_max: int,
        initialization_delay: int,
        pointwise_hit_counter_max: int,
        detection_threshold: float,
        period: int,
        filter_factory: "FilterFactory",
        past_detections_length: int,
        reid_hit_counter_max: int | None,
        coord_transformations: CoordinatesTransformation | None,
    ) -> "TrackedObject":
        """Create a new ``TrackedObject`` and register it with the factory.

        Parameters
        ----------
        initial_detection : Detection
            First detection used to seed the new tracked object.
        hit_counter_max : int
            Maximum value for the hit counter (controls how long an
            unmatched object survives).
        initialization_delay : int
            Number of consecutive hits required before the object is
            considered fully initialized.
        pointwise_hit_counter_max : int
            Maximum hit-counter value tracked per individual point.
        detection_threshold : float
            Confidence threshold below which individual detection points
            are ignored.
        period : int
            Tracker period (frames between ``update`` calls).
        filter_factory : FilterFactory
            Factory used to create the Kalman filter for each point.
        past_detections_length : int
            Maximum number of past detections to store on the object.
        reid_hit_counter_max : int or None
            If not ``None``, maximum hit counter used for re-identification
            of lost objects.
        coord_transformations : CoordinatesTransformation or None
            Optional coordinate transformation for camera-motion
            compensation.

        Returns
        -------
        TrackedObject
            The newly created and registered tracked object.
        """
        obj = TrackedObject(
            obj_factory=self,
            initial_detection=initial_detection,
            hit_counter_max=hit_counter_max,
            initialization_delay=initialization_delay,
            pointwise_hit_counter_max=pointwise_hit_counter_max,
            detection_threshold=detection_threshold,
            period=period,
            filter_factory=filter_factory,
            past_detections_length=past_detections_length,
            reid_hit_counter_max=reid_hit_counter_max,
            coord_transformations=coord_transformations,
        )
        return obj

    def get_initializing_id(self) -> int:
        """Return the next initializing-phase identifier."""
        self.initializing_count += 1
        return self.initializing_count

    def get_ids(self) -> tuple[int, int]:
        """Return a ``(local_id, global_id)`` pair, incrementing both counters."""
        self.count += 1
        with _TrackedObjectFactory._lock:
            _TrackedObjectFactory.global_count += 1
            return self.count, _TrackedObjectFactory.global_count


class TrackedObject:
    """Objects returned by the tracker's ``update`` method on each iteration.

    They represent the objects currently being tracked by the tracker.

    Users should not instantiate ``TrackedObject`` manually; the ``Tracker``
    is in charge of creating them.

    Attributes
    ----------
    estimate : np.ndarray
        Where the tracker predicts the points will be in the current frame
        based on past detections. A NumPy array with the same shape as the
        detections being fed to the tracker that produced it.
    id : int or None
        The unique identifier assigned to this object by the tracker. Set to
        ``None`` while the object is initializing.
    global_id : int or None
        The globally unique identifier assigned to this object. Set to
        ``None`` while the object is initializing.
    last_detection : Detection
        The last detection that matched with this tracked object. Useful if
        you are storing embeddings in your detections and want to do metric
        learning, or for debugging.
    last_distance : float or None
        The distance the tracker had with the last object it matched with.
    age : int
        The age of this object measured in number of frames.
    live_points : np.ndarray
        A boolean mask with shape ``(n_points,)``. Points marked as ``True``
        have recently been matched with detections. Points marked as
        ``False`` are stale and should be ignored.

        Functions like [`draw_points`][norfair.drawing.draw_points] use this
        property to determine which points not to draw.
    initializing_id : int
        On top of ``id``, objects also have an ``initializing_id`` assigned
        internally by the ``Tracker``; this id is used solely for debugging.

        Each new object created by the ``Tracker`` starts as an
        uninitialized ``TrackedObject`` which needs to reach a certain match
        rate before it is converted into a full-fledged tracked object.
        ``initializing_id`` is the id temporarily assigned to a
        ``TrackedObject`` while it is being initialized.

    """

    def __init__(
        self,
        obj_factory: _TrackedObjectFactory,
        initial_detection: "Detection",
        hit_counter_max: int,
        initialization_delay: int,
        pointwise_hit_counter_max: int,
        detection_threshold: float,
        period: int,
        filter_factory: "FilterFactory",
        past_detections_length: int,
        reid_hit_counter_max: int | None,
        coord_transformations: CoordinatesTransformation | None = None,
    ):
        if not isinstance(initial_detection, Detection):
            raise ValueError(
                f"\n[red]ERROR[/red]: The detection list fed into `tracker.update()` should be composed of {Detection} objects not {type(initial_detection)}.\n"
            )
        self._obj_factory = obj_factory
        self.dim_points = initial_detection.absolute_points.shape[1]
        self.num_points = initial_detection.absolute_points.shape[0]
        self.hit_counter_max: int = hit_counter_max
        self.pointwise_hit_counter_max: int = max(pointwise_hit_counter_max, period)
        self.initialization_delay = initialization_delay
        self.detection_threshold: float = detection_threshold
        self.initial_period: int = period
        self.hit_counter: int = period
        self.reid_hit_counter_max = reid_hit_counter_max
        self.reid_hit_counter: int | None = None
        self.last_distance: float | None = None
        self.current_min_distance: float | None = None
        self.last_detection: Detection = copy.copy(initial_detection)
        self.age: int = 0
        self.last_detection.age = self.age
        self.is_initializing: bool = self.hit_counter <= self.initialization_delay
        self.scores = initial_detection.scores

        self.initializing_id: int | None = self._obj_factory.get_initializing_id()
        self.id: int | None = None
        self.global_id: int | None = None
        if not self.is_initializing:
            self._acquire_ids()

        if initial_detection.scores is None:
            self.detected_at_least_once_points = np.array([True] * self.num_points)
        else:
            self.detected_at_least_once_points = (
                initial_detection.scores > self.detection_threshold
            )
        self.point_hit_counter: NDArray[np.intp] = (
            self.detected_at_least_once_points.astype(int)
        )
        # Per-point mask tracking which points were matched in the current frame.
        # On creation, the initial detection counts as the current-frame match.
        # tracker_step() clears this mask before each new frame, and hit() sets
        # it for points that were matched this frame. See GH#2.
        self._point_matched_in_current_frame: NDArray[np.bool_] = (
            self.detected_at_least_once_points.copy()
        )
        self.past_detections_length = past_detections_length
        self.past_detections: list[Detection]
        if past_detections_length > 0:
            stored = copy.copy(initial_detection)
            stored.age = self.age
            self.past_detections = [stored]
        else:
            self.past_detections = []

        # Create Kalman Filter
        self.filter = filter_factory.create_filter(initial_detection.absolute_points)
        self.dim_z = self.dim_points * self.num_points
        self.label = initial_detection.label
        self.abs_to_rel: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = (
            None
        )
        if coord_transformations is not None:
            self.update_coordinate_transformation(coord_transformations)

    def tracker_step(self):
        """Advance the internal state of the tracker by one frame.

        Decrements the hit counters, increments the age and steps the
        Kalman filter prediction forward. Called by the ``Tracker`` once per
        update cycle.
        """
        if self.reid_hit_counter is None:
            if self.hit_counter <= 0:
                self.reid_hit_counter = self.reid_hit_counter_max
        else:
            self.reid_hit_counter -= 1
        self.hit_counter -= 1
        self.point_hit_counter -= 1
        self.age += 1
        # Clear the current-frame match mask. hit() will set entries back to
        # True for points that get matched this frame; points that remain
        # unmatched stay False so live_points correctly reports them as dead.
        self._point_matched_in_current_frame.fill(False)
        # Advances the tracker's state
        self.filter.predict()
        self.scores = None

    @property
    def hit_counter_is_positive(self):
        """Whether the hit counter is still non-negative (object is alive)."""
        return self.hit_counter >= 0

    @property
    def reid_hit_counter_is_positive(self):
        """Whether the ReID hit counter is still non-negative (object can still be re-identified)."""
        return self.reid_hit_counter is None or self.reid_hit_counter >= 0

    @property
    def estimate_velocity(self) -> NDArray[np.float64]:
        """Return the velocity estimate of the object from the Kalman filter.

        The velocity is expressed in the absolute coordinate system.

        Returns
        -------
        np.ndarray
            An array of shape ``(num_points, dim_points)`` containing the
            velocity estimate of the object on each axis.

        """
        return self.filter.x.T.flatten()[self.dim_z :].reshape(-1, self.dim_points)

    @property
    def estimate(self) -> NDArray[np.float64]:
        """Return the position estimate of the object from the Kalman filter.

        Returns
        -------
        NDArray[np.float64]
            An array of shape ``(num_points, dim_points)`` containing the
            position estimate of the object on each axis.

        """
        return self.get_estimate()

    def get_estimate(self, absolute=False) -> NDArray[np.float64]:
        """Return the position estimate in either absolute or relative coordinates.

        Parameters
        ----------
        absolute : bool, optional
            If ``True`` return the estimate in absolute coordinates,
            otherwise return it in the tracker's relative coordinate system.
            Defaults to ``False``.

        Returns
        -------
        NDArray[np.float64]
            An array of shape ``(num_points, dim_points)`` containing the
            position estimate of the object on each axis.

        Raises
        ------
        ValueError
            If absolute coordinates are requested but the tracker has no
            coordinate transformation attached.

        """
        positions = self.filter.x.T.flatten()[: self.dim_z].reshape(-1, self.dim_points)
        if self.abs_to_rel is None:
            if not absolute:
                return positions
            else:
                raise ValueError(
                    "You must provide 'coord_transformations' to the tracker to get absolute coordinates"
                )
        else:
            if absolute:
                return positions
            else:
                return self.abs_to_rel(positions)

    @property
    def live_points(self):
        """Boolean mask marking which tracked points are live in the current frame.

        A point is live only if it was matched to a detection in the
        current frame **and** its pointwise hit counter is still positive.
        """
        return (self.point_hit_counter > 0) & self._point_matched_in_current_frame

    def hit(self, detection: "Detection", period: int = 1):
        """Update this tracked object with a new detection.

        Parameters
        ----------
        detection : Detection
            The new detection matched to this tracked object.
        period : int, optional
            Frames corresponding to the period of time since the last
            update. Defaults to ``1``.

        """
        self._conditionally_add_to_past_detections(detection)

        self.last_detection = copy.copy(detection)
        self.last_detection.age = self.age
        self.hit_counter = min(self.hit_counter + 2 * period, self.hit_counter_max)

        self.scores = detection.scores

        if self.is_initializing and self.hit_counter > self.initialization_delay:
            self.is_initializing = False
            self._acquire_ids()
            self.hit_counter = self.hit_counter_max

        # Reset reid_hit_counter if we are are successfully tracking this object.
        # If hit_counter was 0 when Tracker.update was called and ReID is being used,
        # we preemptively set reid_hit_counter to reid_hit_counter_max. But if the object
        # is hit, we need to reset it.
        if self.hit_counter_is_positive:
            self.reid_hit_counter = None

        # We use a kalman filter in which we consider each coordinate on each point as a sensor.
        # This is a hacky way to update only certain sensors (only x, y coordinates for
        # points which were detected).
        # TODO: Use keypoint confidence information to change R on each sensor instead?
        if detection.scores is not None:
            if len(detection.scores.shape) != 1:
                raise ValueError("Detection scores must be a 1-dimensional array.")
            points_over_threshold_mask = detection.scores > self.detection_threshold
            matched_sensors_mask = np.array(
                [(m,) * self.dim_points for m in points_over_threshold_mask]
            ).flatten()
            H_pos = np.diag(matched_sensors_mask).astype(
                float
            )  # We measure x, y positions
            self.point_hit_counter[points_over_threshold_mask] += 2 * period
        else:
            points_over_threshold_mask = np.array([True] * self.num_points)
            H_pos = np.identity(self.num_points * self.dim_points)
            self.point_hit_counter += 2 * period
        self.point_hit_counter[
            self.point_hit_counter >= self.pointwise_hit_counter_max
        ] = self.pointwise_hit_counter_max
        self.point_hit_counter[self.point_hit_counter < 0] = 0
        # Record which points were matched in the current frame so that
        # `live_points` reflects the match state of *this* frame. See GH#2.
        self._point_matched_in_current_frame = np.logical_or(
            self._point_matched_in_current_frame,
            points_over_threshold_mask,
        )
        H_vel = np.zeros(H_pos.shape)  # But we don't directly measure velocity
        H = np.hstack([H_pos, H_vel])
        self.filter.update(
            np.expand_dims(detection.absolute_points.flatten(), 0).T, None, H
        )

        detected_at_least_once_mask = np.array(
            [(m,) * self.dim_points for m in self.detected_at_least_once_points]
        ).flatten()
        now_detected_mask = np.hstack(
            (points_over_threshold_mask,) * self.dim_points
        ).flatten()
        first_detection_mask = np.logical_and(
            now_detected_mask, np.logical_not(detected_at_least_once_mask)
        )

        self.filter.x[: self.dim_z][first_detection_mask] = np.expand_dims(
            detection.absolute_points.flatten(), 0
        ).T[first_detection_mask]

        # Force points being detected for the first time to have velocity = 0
        # This is needed because some detectors (like OpenPose) set points with
        # low confidence to coordinates (0, 0). And when they then get their first
        # real detection this creates a huge velocity vector in our KalmanFilter
        # and causes the tracker to start with wildly inaccurate estimations which
        # eventually coverge to the real detections.
        self.filter.x[self.dim_z :][np.logical_not(detected_at_least_once_mask)] = 0
        self.detected_at_least_once_points = np.logical_or(
            self.detected_at_least_once_points, points_over_threshold_mask
        )

    def __repr__(self):
        """Return a human-readable representation of this tracked object."""
        if self.last_distance is None:
            placeholder_text = "\033[1mObject_{}\033[0m(age: {}, hit_counter: {}, last_distance: {}, init_id: {})"
        else:
            placeholder_text = "\033[1mObject_{}\033[0m(age: {}, hit_counter: {}, last_distance: {:.2f}, init_id: {})"
        return placeholder_text.format(
            self.id,
            self.age,
            self.hit_counter,
            self.last_distance,
            self.initializing_id,
        )

    def _conditionally_add_to_past_detections(self, detection):
        """Maintain the rolling buffer of past detections for this object.

        Keeps a fixed number of past detections saved in each
        ``TrackedObject`` while distributing them uniformly through the
        object's lifetime.

        The detection is shallow-copied before storage so the caller's
        original ``Detection`` object is never mutated.
        """
        if self.past_detections_length == 0:
            return
        stored = copy.copy(detection)
        stored.age = self.age
        if len(self.past_detections) < self.past_detections_length:
            self.past_detections.append(stored)
        else:
            first_detection_age = self.past_detections[0].age
            if (
                first_detection_age is not None
                and self.age >= first_detection_age * self.past_detections_length
            ):
                self.past_detections.pop(0)
                self.past_detections.append(stored)

    def merge(self, tracked_object):
        """Merge another (not-yet-initialized) ``TrackedObject`` into this one.

        Mutable state (filter, counters, arrays) is copied so the
        discarded ``tracked_object`` is not silently aliased.

        Parameters
        ----------
        tracked_object : TrackedObject
            Another tracked object (typically not yet fully initialized)
            whose state will be absorbed into this one.
        """
        self.reid_hit_counter = None
        self.hit_counter = self.initial_period * 2
        self.point_hit_counter = tracked_object.point_hit_counter.copy()
        # The not-yet-initialized tracked object was just matched this frame,
        # so adopt its current-frame match mask as well (GH#2).
        self._point_matched_in_current_frame = (
            tracked_object._point_matched_in_current_frame.copy()
        )
        self.last_distance = tracked_object.last_distance
        self.current_min_distance = tracked_object.current_min_distance
        self.last_detection = copy.copy(tracked_object.last_detection)
        self.last_detection.age = self.age
        self.detected_at_least_once_points = (
            tracked_object.detected_at_least_once_points.copy()
        )
        self.filter = copy.deepcopy(tracked_object.filter)

        for past_detection in tracked_object.past_detections:
            self._conditionally_add_to_past_detections(past_detection)

    def update_coordinate_transformation(
        self, coordinate_transformation: CoordinatesTransformation | None
    ):
        """Attach (or refresh) the abs→rel converter from a new coordinate transformation.

        Parameters
        ----------
        coordinate_transformation : CoordinatesTransformation or None
            The new coordinate transformation whose ``abs_to_rel`` method
            will be stored. If ``None``, the existing converter is left
            unchanged.
        """
        if coordinate_transformation is not None:
            self.abs_to_rel = coordinate_transformation.abs_to_rel

    def _acquire_ids(self):
        """Assign a concrete ``id`` and ``global_id`` from the factory."""
        self.id, self.global_id = self._obj_factory.get_ids()


class Detection:
    """A single detection produced by a detector, prepared for use by Norfair.

    Detections returned by the detector must be converted to a ``Detection``
    object before being passed to the ``Tracker``.

    Parameters
    ----------
    points : np.ndarray
        Points detected. Must be a rank-2 array with shape
        ``(n_points, n_dimensions)``.
    scores : np.ndarray or float, optional
        A score per point, or a single scalar score applied to every point.
        When an array is given, its length must match ``n_points``.

        This is used to inform the tracker which points to ignore: any
        point with a score at or below ``detection_threshold`` is skipped.

        Useful when detections don't always have every point present, as
        is often the case in pose estimators.
    data : Any, optional
        A place to store any extra data which may be useful when calculating
        the distance function. Anything stored here will be available to use
        inside the distance function.

        This enables more interesting trackers that, for instance, attach an
        appearance embedding to each detection to aid in tracking.
    label : Hashable, optional
        When working with multiple classes, the detection's label can be
        stored to be used as a matching condition when associating tracked
        objects with new detections. The label's type must be hashable for
        drawing purposes.
    embedding : Any, optional
        An embedding used by the ReID distance function, if any.

    Attributes
    ----------
    points : np.ndarray
        The detection points, validated to shape ``(n_points, n_dimensions)``.
    absolute_points : np.ndarray
        Starts as a copy of ``points``. When ``coord_transformations`` is
        passed to [`Tracker.update`][norfair.tracker.Tracker.update], this
        is overwritten with absolute coordinates.
    age : int or None
        Set by the tracker to the matched ``TrackedObject``'s age when the
        detection is stored as a past detection. ``None`` until then.

    """

    def __init__(
        self,
        points: NDArray[np.float64],
        scores: float | int | NDArray[np.float64] | None = None,
        data: Any = None,
        label: Hashable = None,
        embedding: Any = None,
    ):
        self.points = validate_points(points)

        self.scores: NDArray[np.float64] | None
        if isinstance(scores, np.ndarray):
            if len(scores) != len(self.points):
                raise ValueError(
                    f"scores should be a np.ndarray with length {len(self.points)}, but got length {len(scores)}."
                )
            self.scores = scores
        elif scores is not None:
            self.scores = np.zeros((len(points),)) + scores
        else:
            self.scores = None
        self.data = data
        self.label = label
        self.absolute_points = self.points.copy()
        self.embedding = embedding
        self.age: int | None = None

    def update_coordinate_transformation(
        self, coordinate_transformation: CoordinatesTransformation
    ):
        """Rewrite ``absolute_points`` into absolute coordinates.

        Parameters
        ----------
        coordinate_transformation : CoordinatesTransformation
            Transformation used to map relative points into the absolute
            reference frame. When ``None``, this is a no-op.

        """
        if coordinate_transformation is not None:
            self.absolute_points = coordinate_transformation.rel_to_abs(
                self.absolute_points
            )
