import numpy as np

from norfair.camera_motion import HomographyTransformation


def test_homography_1d_point():
    """Test that HomographyTransformation handles 1D point arrays without crashing."""
    # Identity homography — points should pass through unchanged
    H = np.eye(3)
    transform = HomographyTransformation(H)

    point_1d = np.array([100.0, 200.0])
    point_2d = np.array([[100.0, 200.0]])

    # rel_to_abs with 1D input
    result_1d = transform.rel_to_abs(point_1d)
    assert result_1d.ndim == 1
    np.testing.assert_allclose(result_1d, point_1d)

    # rel_to_abs with 2D input (should still work as before)
    result_2d = transform.rel_to_abs(point_2d)
    assert result_2d.ndim == 2
    np.testing.assert_allclose(result_2d, point_2d)

    # abs_to_rel with 1D input
    result_1d = transform.abs_to_rel(point_1d)
    assert result_1d.ndim == 1
    np.testing.assert_allclose(result_1d, point_1d)

    # abs_to_rel with 2D input (should still work as before)
    result_2d = transform.abs_to_rel(point_2d)
    assert result_2d.ndim == 2
    np.testing.assert_allclose(result_2d, point_2d)


def test_homography_1d_non_identity():
    """Test 1D input with a non-identity homography produces correct results."""
    # Translation homography: shifts x by +10, y by +20
    H = np.array(
        [
            [1, 0, 10],
            [0, 1, 20],
            [0, 0, 1],
        ],
        dtype=float,
    )
    transform = HomographyTransformation(H)

    point_1d = np.array([100.0, 200.0])

    # rel_to_abs uses inverse homography
    result = transform.rel_to_abs(point_1d)
    assert result.ndim == 1
    np.testing.assert_allclose(result, np.array([90.0, 180.0]))

    # abs_to_rel uses forward homography
    result = transform.abs_to_rel(point_1d)
    assert result.ndim == 1
    np.testing.assert_allclose(result, np.array([110.0, 220.0]))
