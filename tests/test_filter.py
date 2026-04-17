import numpy as np

from norfair.filter import NoFilter, NoFilterFactory, OptimizedKalmanFilterFactory


def test_optimized_filter_vel_variance_stays_non_negative():
    """``vel_variance`` must stay non-negative after an update.

    A large ``pos_vel_covariance`` paired with a small ``vel_variance``
    subtracts a large term from the running variance; numerical
    cancellation would otherwise drive it below zero.
    """
    factory = OptimizedKalmanFilterFactory(
        R=0.01,
        Q=0.001,
        pos_variance=0.1,
        pos_vel_covariance=50.0,
        vel_variance=0.01,
    )
    initial_detection = np.array([[0.0, 0.0]])
    f = factory.create_filter(initial_detection)

    f.predict()
    measurement = np.array([[1.0], [1.0]])
    f.update(measurement)

    assert np.all(f.vel_variance >= 0), (
        f"vel_variance went negative: {f.vel_variance.flatten()}"
    )


class TestNoFilterFactory:
    def test_create(self):
        factory = NoFilterFactory()
        initial_detection = np.array([[1.0, 2.0], [3.0, 4.0]])
        f = factory.create_filter(initial_detection)
        assert isinstance(f, NoFilter)
        # Check initial state matches detection
        dim_z = 4  # 2 points * 2 dims
        np.testing.assert_array_equal(
            f.x[:dim_z].flatten(), initial_detection.flatten()
        )

    def test_predict_is_noop(self):
        factory = NoFilterFactory()
        initial_detection = np.array([[1.0, 2.0]])
        f = factory.create_filter(initial_detection)
        x_before = f.x.copy()
        f.predict()
        np.testing.assert_array_equal(f.x, x_before)

    def test_update_replaces_position(self):
        factory = NoFilterFactory()
        initial_detection = np.array([[1.0, 2.0]])
        f = factory.create_filter(initial_detection)

        new_points = np.array([[5.0, 6.0]])
        f.update(np.expand_dims(new_points.flatten(), 0).T)
        dim_z = 2
        np.testing.assert_array_equal(f.x[:dim_z].flatten(), new_points.flatten())

    def test_update_with_partial_H(self):
        factory = NoFilterFactory()
        initial_detection = np.array([[1.0, 2.0], [3.0, 4.0]])
        f = factory.create_filter(initial_detection)

        # Only update first point (mask second point)
        H_pos = np.diag([1, 1, 0, 0]).astype(float)
        new_points = np.array([[10.0, 20.0], [30.0, 40.0]])
        f.update(np.expand_dims(new_points.flatten(), 0).T, H=H_pos)

        # First point should be updated
        np.testing.assert_almost_equal(f.x[0, 0], 10.0)
        np.testing.assert_almost_equal(f.x[1, 0], 20.0)
        # Second point should keep original values
        np.testing.assert_almost_equal(f.x[2, 0], 3.0)
        np.testing.assert_almost_equal(f.x[3, 0], 4.0)
