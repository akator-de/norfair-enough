"""Predictive filter factories used by ``Tracker`` to estimate object motion."""

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from .kalman_filter import KalmanFilter


class Filter(Protocol):
    """Protocol defining the interface for prediction filters."""

    x: np.ndarray

    def predict(self) -> None:
        """Advance the internal state one step forward in time."""
        ...

    def update(
        self,
        __z: np.ndarray,
        /,
        R: np.ndarray | None = None,
        H: np.ndarray | None = None,
    ) -> None:
        """Incorporate a new measurement ``z`` into the state."""
        ...


class FilterFactory(ABC):
    """Abstract base class for predictive-filter factories.

    Subclasses must implement :meth:`create_filter`, which returns a new
    filter instance for each tracked object.
    """

    @abstractmethod
    def create_filter(self, initial_detection: np.ndarray) -> Filter:
        """Return a new filter seeded with ``initial_detection``."""
        ...


class FilterPyKalmanFilterFactory(FilterFactory):
    """Factory for filterpy-backed Kalman filters.

    Use this factory to either tweak the parameters of the
    [KalmanFilter](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html)
    that the tracker uses, or to fully customize the predictive filter
    implementation (as long as the methods and properties are compatible).

    In the first case, only the default parameters need to be tweaked at
    tracker creation time::

        tracker = Tracker(..., filter_factory=FilterPyKalmanFilterFactory(R=100))

    In the second case, create your own subclass of
    ``FilterPyKalmanFilterFactory`` and override :meth:`create_filter` to
    return your customized filter.

    Parameters
    ----------
    R : float, optional
        Multiplier for the sensor measurement noise matrix. Defaults to
        ``4.0``. Larger values make the filter trust measurements less
        and rely more on its own predictions — useful when detections
        are noisy.
    Q : float, optional
        Multiplier for the process uncertainty. Defaults to ``0.1``.
        Larger values let the filter react faster to real motion changes
        but increase jitter; smaller values produce smoother but laggier
        tracks.
    P : float, optional
        Multiplier for the initial covariance matrix estimation, only in
        the entries that correspond to position (not speed) variables.
        Defaults to ``10.0``.

    Notes
    -----
    The generated Kalman filter uses a state vector of length ``2 * dim_z``
    laid out as ``[pos[0], ..., pos[dim_z-1], vel[0], ..., vel[dim_z-1]]``,
    where ``dim_z = num_points * dim_points`` is the flattened measurement
    dimensionality. This layout is assumed by the rest of the tracker and
    should be preserved when subclassing.

    When the innovation covariance matrix ``S`` is singular or
    ill-conditioned (condition number > ``1e12``), the filter automatically
    falls back to a pseudo-inverse to maintain numerical stability.

    See Also
    --------
    [`filterpy.KalmanFilter`](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html)

    """

    def __init__(self, R: float = 4.0, Q: float = 0.1, P: float = 10.0):
        self.R = R
        self.Q = Q
        self.P = P

    def create_filter(self, initial_detection: np.ndarray) -> KalmanFilter:
        """Return a new Kalman filter seeded with ``initial_detection``.

        The returned filter is used by each new
        [`TrackedObject`][norfair.tracker.TrackedObject] to estimate speed
        and future positions so detections can be matched along the
        trajectory.

        Parameters
        ----------
        initial_detection : np.ndarray
            Array of shape ``(n_points, n_dimensions)`` corresponding to
            [`Detection.points`][norfair.tracker.Detection] of the tracked
            object being born, used as the initial position estimate.

        Returns
        -------
        KalmanFilter
            A freshly initialized Kalman filter.

        """
        num_points = initial_detection.shape[0]
        dim_points = initial_detection.shape[1]
        dim_z = dim_points * num_points
        dim_x = 2 * dim_z  # We need to accommodate for velocities

        filter = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

        # State transition matrix (models physics): numpy.array()
        filter.F = np.eye(dim_x)
        dt = 1  # At each step we update pos with v * dt

        filter.F[:dim_z, dim_z:] = dt * np.eye(dim_z)

        # Measurement function: numpy.array(dim_z, dim_x)
        filter.H = np.eye(
            dim_z,
            dim_x,
        )

        # Measurement uncertainty (sensor noise): numpy.array(dim_z, dim_z)
        filter.R *= self.R

        # Process uncertainty: numpy.array(dim_x, dim_x)
        # Don't decrease it too much or trackers pay too little attention to detections
        filter.Q[dim_z:, dim_z:] *= self.Q

        # Initial state: numpy.array(dim_x, 1)
        filter.x[:dim_z] = np.expand_dims(initial_detection.flatten(), 0).T
        filter.x[dim_z:] = 0

        # Estimation uncertainty: numpy.array(dim_x, dim_x)
        filter.P[dim_z:, dim_z:] *= self.P

        return filter


class NoFilter:
    """Null filter that keeps the last observation as its state.

    Used by :class:`NoFilterFactory` to disable predictive filtering.
    """

    def __init__(self, dim_x, dim_z):
        """Initialize the null filter with a zero state vector of length ``dim_x``."""
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))

    def predict(self):
        """No-op predict step — the state does not evolve between updates."""
        return

    def update(self, detection_points_flatten, R=None, H=None):
        """Overwrite the position portion of the state with the new detection.

        Parameters
        ----------
        detection_points_flatten : np.ndarray
            Column vector of flattened detection points.
        R : np.ndarray, optional
            Ignored. Kept for API compatibility with filterpy filters.
        H : np.ndarray, optional
            Measurement function. Only its diagonal is used, to mask out
            points that were not observed in the current frame.

        """
        if H is not None:
            diagonal = np.diagonal(H).reshape((self.dim_z, 1))
            one_minus_diagonal = 1 - diagonal

            detection_points_flatten = np.multiply(
                diagonal, detection_points_flatten
            ) + np.multiply(one_minus_diagonal, self.x[: self.dim_z])

        self.x[: self.dim_z] = detection_points_flatten


class NoFilterFactory(FilterFactory):
    """Factory producing a null filter with no velocity estimation.

    Lets the user try Norfair without any predictive filtering: tracking is
    performed only by comparing the position of previous detections to
    those in the current frame.

    The throughput of this class in FPS is similar to that of
    [`OptimizedKalmanFilterFactory`][norfair.filter.OptimizedKalmanFilterFactory],
    so this class exists only for comparative purposes and is not advised
    for real-world tracking.
    """

    def create_filter(self, initial_detection: np.ndarray):
        """Return a :class:`NoFilter` seeded with ``initial_detection``."""
        num_points = initial_detection.shape[0]
        dim_points = initial_detection.shape[1]
        dim_z = dim_points * num_points  # flattened positions
        dim_x = 2 * dim_z  # We need to accommodate for velocities

        no_filter = NoFilter(
            dim_x,
            dim_z,
        )
        no_filter.x[:dim_z] = np.expand_dims(initial_detection.flatten(), 0).T
        return no_filter


class OptimizedKalmanFilter:
    """A Kalman filter specialized and vectorized for Norfair tracking.

    This implementation exploits the structural properties of Norfair's
    tracking problem (diagonal covariance, independent axes) to be faster
    than the generic ``filterpy`` implementation.
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        pos_variance: float = 10.0,
        pos_vel_covariance: float = 0.0,
        vel_variance: float = 1.0,
        q: float = 0.1,
        r: float = 4.0,
    ):
        """Initialize the filter state and per-axis variance buffers."""
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))

        # matrix P from Kalman
        self.pos_variance = np.zeros((dim_z, 1)) + pos_variance
        self.pos_vel_covariance = np.zeros((dim_z, 1)) + pos_vel_covariance
        self.vel_variance = np.zeros((dim_z, 1)) + vel_variance

        self.q_Q = q

        self.default_r = r * np.ones((dim_z, 1))

    def predict(self):
        """Advance positions by the current velocity estimate."""
        self.x[: self.dim_z] += self.x[self.dim_z :]

    def update(self, detection_points_flatten, R=None, H=None):
        """Fold a new measurement into the filter state.

        Parameters
        ----------
        detection_points_flatten : np.ndarray
            Column vector of flattened detection points.
        R : np.ndarray, optional
            Measurement noise matrix. Only the diagonal is used; when
            ``None``, falls back to the factory default.
        H : np.ndarray, optional
            Measurement function. Only the diagonal is used, to select
            which points were actually observed.

        """
        if H is not None:
            diagonal = np.diagonal(H).reshape((self.dim_z, 1))
            one_minus_diagonal = 1 - diagonal
        else:
            diagonal = np.ones((self.dim_z, 1))
            one_minus_diagonal = np.zeros((self.dim_z, 1))

        if R is not None:
            kalman_r = np.diagonal(R).reshape((self.dim_z, 1))
        else:
            kalman_r = self.default_r

        error = np.multiply(detection_points_flatten - self.x[: self.dim_z], diagonal)

        vel_var_plus_pos_vel_cov = self.pos_vel_covariance + self.vel_variance
        added_variances = (
            self.pos_variance
            + self.pos_vel_covariance
            + vel_var_plus_pos_vel_cov
            + self.q_Q
            + kalman_r
        )
        # Guard against zero-variance division (see issue #46).
        added_variances = np.maximum(added_variances, 1e-12)

        kalman_r_over_added_variances = np.divide(kalman_r, added_variances)
        vel_var_plus_pos_vel_cov_over_added_variances = np.divide(
            vel_var_plus_pos_vel_cov, added_variances
        )

        added_variances_or_kalman_r = np.multiply(
            added_variances, one_minus_diagonal
        ) + np.multiply(kalman_r, diagonal)

        self.x[: self.dim_z] += np.multiply(
            diagonal, np.multiply(1 - kalman_r_over_added_variances, error)
        )
        self.x[self.dim_z :] += np.multiply(
            diagonal, np.multiply(vel_var_plus_pos_vel_cov_over_added_variances, error)
        )

        self.pos_variance = np.multiply(
            1 - kalman_r_over_added_variances, added_variances_or_kalman_r
        )
        self.pos_vel_covariance = np.multiply(
            vel_var_plus_pos_vel_cov_over_added_variances, added_variances_or_kalman_r
        )
        self.vel_variance += self.q_Q - np.multiply(
            diagonal,
            np.multiply(
                np.square(vel_var_plus_pos_vel_cov_over_added_variances),
                added_variances,
            ),
        )


class OptimizedKalmanFilterFactory(FilterFactory):
    """Factory for the vectorized :class:`OptimizedKalmanFilter`.

    Produces filters that are faster than those returned by
    [`FilterPyKalmanFilterFactory`][norfair.filter.FilterPyKalmanFilterFactory]
    and exposes the most relevant tuning knobs.

    Parameters
    ----------
    R : float, optional
        Multiplier for the sensor measurement noise matrix. Larger values
        make the filter trust measurements less — useful when detections
        are noisy.
    Q : float, optional
        Multiplier for the process uncertainty. Larger values let the filter
        react faster to real motion changes but increase jitter; smaller
        values produce smoother but laggier tracks.
    pos_variance : float, optional
        Multiplier for the initial covariance matrix estimation in the
        entries that correspond to position (not speed) variables.
    pos_vel_covariance : float, optional
        Multiplier for the initial covariance matrix estimation in the
        entries that correspond to the covariance between position and
        velocity.
    vel_variance : float, optional
        Multiplier for the initial covariance matrix estimation in the
        entries that correspond to velocity (not position) variables.

    Notes
    -----
    The generated filter uses the same state vector layout as
    [`FilterPyKalmanFilterFactory`][norfair.filter.FilterPyKalmanFilterFactory]:
    ``[pos[0], ..., pos[dim_z-1], vel[0], ..., vel[dim_z-1]]``, with
    ``dim_z = num_points * dim_points``.

    Intermediate variances are clamped to a minimum of ``1e-12`` to prevent
    zero-division in degenerate cases (e.g. perfectly stationary targets with
    near-zero measurement noise).
    """

    def __init__(
        self,
        R: float = 4.0,
        Q: float = 0.1,
        pos_variance: float = 10,
        pos_vel_covariance: float = 0,
        vel_variance: float = 1,
    ):
        """Store the factory-wide tuning parameters."""
        self.R = R
        self.Q = Q

        # entrances P matrix of KF
        self.pos_variance = pos_variance
        self.pos_vel_covariance = pos_vel_covariance
        self.vel_variance = vel_variance

    def create_filter(self, initial_detection: np.ndarray):
        """Return an :class:`OptimizedKalmanFilter` seeded with ``initial_detection``."""
        num_points = initial_detection.shape[0]
        dim_points = initial_detection.shape[1]
        dim_z = dim_points * num_points  # flattened positions
        dim_x = 2 * dim_z  # We need to accommodate for velocities

        custom_filter = OptimizedKalmanFilter(
            dim_x,
            dim_z,
            pos_variance=self.pos_variance,
            pos_vel_covariance=self.pos_vel_covariance,
            vel_variance=self.vel_variance,
            q=self.Q,
            r=self.R,
        )
        custom_filter.x[:dim_z] = np.expand_dims(initial_detection.flatten(), 0).T

        return custom_filter
