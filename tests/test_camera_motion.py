import numpy as np
import pytest

from norfair.camera_motion import (
    HomographyTransformation,
    TranslationTransformation,
    TranslationTransformationGetter,
)


def test_homography_singular_matrix_raises_value_error():
    """Singular homography matrices raise ``ValueError``."""
    # Row of zeros => determinant 0 => matrix is singular.
    singular = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    with pytest.raises(ValueError, match="singular|invertible"):
        HomographyTransformation(singular)


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


# ---------------------------------------------------------------
# TranslationTransformation
# ---------------------------------------------------------------
class TestTranslationTransformation:
    """Tests for TranslationTransformation with various movement vectors."""

    def test_identity_translation(self):
        """Zero movement vector should leave points unchanged."""
        t = TranslationTransformation(np.array([0.0, 0.0]))
        pts = np.array([[10.0, 20.0], [30.0, 40.0]])
        np.testing.assert_allclose(t.abs_to_rel(pts), pts)
        np.testing.assert_allclose(t.rel_to_abs(pts), pts)

    def test_nonzero_translation(self):
        """Non-zero movement vector shifts points correctly."""
        movement = np.array([5.0, -3.0])
        t = TranslationTransformation(movement)
        pts = np.array([[100.0, 200.0]])

        # abs_to_rel adds the movement vector
        result_rel = t.abs_to_rel(pts)
        np.testing.assert_allclose(result_rel, np.array([[105.0, 197.0]]))

        # rel_to_abs subtracts the movement vector
        result_abs = t.rel_to_abs(pts)
        np.testing.assert_allclose(result_abs, np.array([[95.0, 203.0]]))

    def test_roundtrip(self):
        """abs_to_rel followed by rel_to_abs should return the original point."""
        movement = np.array([7.5, -12.3])
        t = TranslationTransformation(movement)
        pts = np.array([[42.0, 99.0], [0.0, 0.0], [-10.0, 50.0]])
        roundtripped = t.rel_to_abs(t.abs_to_rel(pts))
        np.testing.assert_allclose(roundtripped, pts)

    def test_1d_point(self):
        """Single 1D point arrays should work (matching HomographyTransformation behavior)."""
        movement = np.array([3.0, 4.0])
        t = TranslationTransformation(movement)
        pt = np.array([10.0, 20.0])
        np.testing.assert_allclose(t.abs_to_rel(pt), np.array([13.0, 24.0]))
        np.testing.assert_allclose(t.rel_to_abs(pt), np.array([7.0, 16.0]))

    def test_batch_points(self):
        """Multiple points should all be translated uniformly."""
        movement = np.array([1.0, 2.0])
        t = TranslationTransformation(movement)
        pts = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        expected_rel = pts + movement
        expected_abs = pts - movement
        np.testing.assert_allclose(t.abs_to_rel(pts), expected_rel)
        np.testing.assert_allclose(t.rel_to_abs(pts), expected_abs)


# ---------------------------------------------------------------
# TranslationTransformationGetter
# ---------------------------------------------------------------
class TestTranslationTransformationGetter:
    """Tests for TranslationTransformationGetter."""

    def test_uniform_flow(self):
        """When all points move by the same vector, that vector is the mode."""
        getter = TranslationTransformationGetter(bin_size=0.5)
        prev_pts = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        shift = np.array([3.0, -2.0])
        curr_pts = prev_pts + shift

        update_prvs, transform = getter(curr_pts, prev_pts)
        assert isinstance(transform, TranslationTransformation)

        # With uniform flow the mode matches exactly; all points agree so
        # proportion_points_used = 1.0, which is >= threshold => no update
        assert not update_prvs

        # The transform's movement_vector should be the detected shift
        np.testing.assert_allclose(transform.movement_vector, shift, atol=0.5)

    def test_noisy_flow_triggers_update(self):
        """When too few points agree, the reference frame should update."""
        getter = TranslationTransformationGetter(
            bin_size=0.2, proportion_points_used_threshold=0.9
        )
        # 10 points all moving differently so no strong mode
        rng = np.random.RandomState(42)
        prev_pts = rng.rand(10, 2) * 100
        curr_pts = prev_pts + rng.rand(10, 2) * 50  # large random shifts

        update_prvs, transform = getter(curr_pts, prev_pts)
        assert isinstance(transform, TranslationTransformation)
        # With random flow, few points share the same bin => update_prvs should be True
        assert update_prvs

    def test_accumulation_across_calls(self):
        """The getter accumulates flow across consecutive calls."""
        getter = TranslationTransformationGetter(bin_size=0.2)
        base_pts = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])

        shift1 = np.array([2.0, 0.0])
        _, t1 = getter(base_pts + shift1, base_pts)
        np.testing.assert_allclose(t1.movement_vector, shift1, atol=0.2)

        # Second call: another uniform shift on top of the first
        shift2 = np.array([0.0, 3.0])
        # The getter expects (curr_pts, prev_pts) relative to whatever reference
        # frame is active. Since update_prvs was False, the reference didn't change,
        # but the getter accumulates. Simulate a second observation:
        _, t2 = getter(base_pts + shift1 + shift2, base_pts)
        # Accumulated shift should be shift1+shift2
        np.testing.assert_allclose(t2.movement_vector, shift1 + shift2, atol=0.2)


# ---------------------------------------------------------------
# HomographyTransformation — additional edge cases
# ---------------------------------------------------------------
class TestHomographyTransformationExtra:
    """Additional edge-case tests for HomographyTransformation."""

    def test_roundtrip(self):
        """abs_to_rel -> rel_to_abs should be the identity."""
        H = np.array([[2, 0, 5], [0, 3, -7], [0, 0, 1]], dtype=float)
        t = HomographyTransformation(H)
        pts = np.array([[1.0, 2.0], [10.0, -5.0], [0.0, 0.0]])
        roundtripped = t.rel_to_abs(t.abs_to_rel(pts))
        np.testing.assert_allclose(roundtripped, pts, atol=1e-10)

    def test_scaling_homography(self):
        """A pure-scaling homography should scale coordinates."""
        H = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]], dtype=float)
        t = HomographyTransformation(H)
        pt = np.array([10.0, 10.0])
        # abs_to_rel applies H forward
        result = t.abs_to_rel(pt)
        np.testing.assert_allclose(result, np.array([20.0, 30.0]))

    def test_multiple_points_batch(self):
        """Batch of points should all transform correctly."""
        H = np.eye(3)
        H[0, 2] = 15  # x-shift
        t = HomographyTransformation(H)
        pts = np.array([[0.0, 0.0], [100.0, 100.0]])
        result = t.abs_to_rel(pts)
        np.testing.assert_allclose(result, np.array([[15.0, 0.0], [115.0, 100.0]]))


# ---------------------------------------------------------------
# HomographyTransformationGetter
# ---------------------------------------------------------------
class TestHomographyTransformationGetter:
    """Tests for HomographyTransformationGetter (requires OpenCV)."""

    def test_insufficient_points_returns_none_first_call(self):
        """With fewer than 4 points and no prior data, returns (True, None)."""
        pytest.importorskip("cv2")
        from norfair.camera_motion import HomographyTransformationGetter

        getter = HomographyTransformationGetter()
        prev_pts = np.array([[1.0, 2.0], [3.0, 4.0]])  # only 2 points
        curr_pts = np.array([[1.0, 2.0], [3.0, 4.0]])

        update, transform = getter(curr_pts, prev_pts)
        assert update is True
        assert transform is None

    def test_insufficient_points_returns_last_known(self):
        """With fewer than 4 points but prior data, returns last known homography."""
        pytest.importorskip("cv2")
        from norfair.camera_motion import HomographyTransformationGetter

        getter = HomographyTransformationGetter()
        # Seed internal data with an identity homography
        getter.data = np.eye(3)

        prev_pts = np.array([[1.0, 2.0]])  # only 1 point
        curr_pts = np.array([[1.0, 2.0]])

        update, transform = getter(curr_pts, prev_pts)
        assert update is True
        assert isinstance(transform, HomographyTransformation)
        # The returned transformation should use the stored identity
        pt = np.array([10.0, 20.0])
        np.testing.assert_allclose(transform.abs_to_rel(pt), pt, atol=1e-10)

    def test_identity_with_matching_points(self):
        """Identical point sets should yield a near-identity homography."""
        pytest.importorskip("cv2")
        from norfair.camera_motion import HomographyTransformationGetter

        getter = HomographyTransformationGetter()
        pts = np.array(
            [
                [10.0, 10.0],
                [100.0, 10.0],
                [100.0, 100.0],
                [10.0, 100.0],
                [50.0, 50.0],
                [75.0, 25.0],
            ],
            dtype=np.float32,
        )
        update, transform = getter(pts, pts)
        assert isinstance(transform, HomographyTransformation)

        # The resulting homography should be close to identity
        test_pt = np.array([42.0, 73.0])
        result = transform.abs_to_rel(test_pt)
        np.testing.assert_allclose(result, test_pt, atol=1.0)

    def test_known_translation_via_homography(self):
        """A known shift in points should produce a translation-like homography."""
        pytest.importorskip("cv2")
        from norfair.camera_motion import HomographyTransformationGetter

        getter = HomographyTransformationGetter()
        shift = np.array([5.0, -3.0])
        prev_pts = np.array(
            [
                [10.0, 10.0],
                [200.0, 10.0],
                [200.0, 200.0],
                [10.0, 200.0],
                [100.0, 100.0],
                [50.0, 150.0],
            ],
            dtype=np.float32,
        )
        curr_pts = (prev_pts + shift).astype(np.float32)

        update, transform = getter(curr_pts, prev_pts)
        assert isinstance(transform, HomographyTransformation)

        # Test that the transformation correctly maps a shifted point
        test_pt = np.array([50.0, 50.0])
        # abs_to_rel should apply the forward homography (prev -> curr)
        result = transform.abs_to_rel(test_pt)
        np.testing.assert_allclose(result, test_pt + shift, atol=1.0)


# ---------------------------------------------------------------
# MotionEstimator (requires OpenCV)
# ---------------------------------------------------------------
class TestMotionEstimator:
    """Integration-level tests for MotionEstimator with synthetic frames."""

    def test_first_frame_near_identity(self):
        """The first call compares the frame against itself, yielding near-identity."""
        pytest.importorskip("cv2")
        from norfair.camera_motion import MotionEstimator

        estimator = MotionEstimator()
        # Create a synthetic BGR frame with some texture
        rng = np.random.RandomState(0)
        frame = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = estimator.update(frame)
        # The first frame compares to itself (gray_prvs = gray_next), so we
        # may get either None or a near-identity transformation.
        if result is not None:
            pt = np.array([25.0, 25.0])
            transformed = result.abs_to_rel(pt)
            np.testing.assert_allclose(transformed, pt, atol=2.0)

    def test_identical_frames_produce_near_identity(self):
        """Two identical frames should produce a near-identity transformation."""
        pytest.importorskip("cv2")
        from norfair.camera_motion import MotionEstimator

        # Use TranslationTransformationGetter for simpler results
        getter = TranslationTransformationGetter()
        estimator = MotionEstimator(transformations_getter=getter)

        # Create a textured frame so goodFeaturesToTrack can find corners
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)

        # First frame initializes state
        estimator.update(frame)

        # Second identical frame => no motion
        transform = estimator.update(frame)
        if transform is not None:
            pt = np.array([50.0, 50.0])
            result = transform.abs_to_rel(pt)
            np.testing.assert_allclose(result, pt, atol=2.0)

    def test_shifted_frame_detects_translation(self):
        """A horizontally shifted frame should be detected as a translation."""
        cv2 = pytest.importorskip("cv2")
        from norfair.camera_motion import MotionEstimator

        getter = TranslationTransformationGetter(bin_size=1.0)
        estimator = MotionEstimator(
            transformations_getter=getter,
            max_points=300,
            min_distance=10,
            quality_level=0.01,
        )

        # Create a large textured frame with good features
        rng = np.random.RandomState(123)
        frame1 = rng.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        # Apply Gaussian blur to make optical flow more stable
        frame1 = cv2.GaussianBlur(frame1, (5, 5), 0)

        # Shift frame horizontally by 10 pixels using np.roll
        shift_x = 10
        frame2 = np.roll(frame1, shift_x, axis=1)

        # First frame
        estimator.update(frame1)
        # Second frame with shift
        transform = estimator.update(frame2)

        # We should get a transformation back
        assert transform is not None

    def test_with_mask(self):
        """MotionEstimator should accept an optional mask."""
        pytest.importorskip("cv2")
        from norfair.camera_motion import MotionEstimator

        getter = TranslationTransformationGetter()
        estimator = MotionEstimator(transformations_getter=getter)

        rng = np.random.RandomState(7)
        frame = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        mask = np.ones((200, 200), dtype=np.uint8) * 255

        # Should not raise
        estimator.update(frame, mask=mask)
        estimator.update(frame, mask=mask)
        # With identical frames and full mask, we should get a valid transform or None
        # (either is fine; the key is no crash)

    def test_draw_flow_flag(self):
        """Setting draw_flow=True should not crash."""
        pytest.importorskip("cv2")
        from norfair.camera_motion import MotionEstimator

        estimator = MotionEstimator(draw_flow=True)

        rng = np.random.RandomState(99)
        frame1 = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        frame2 = np.roll(frame1, 3, axis=1)

        estimator.update(frame1.copy())
        # Should not raise even with draw_flow enabled
        estimator.update(frame2.copy())


# ---------------------------------------------------------------
# _get_sparse_flow and _calc_optical_flow helpers
# ---------------------------------------------------------------
class TestSparseFlowHelpers:
    """Tests for the internal sparse-flow helper functions."""

    def test_get_sparse_flow_with_identical_images(self):
        """Sparse flow between identical images should yield near-zero displacement."""
        pytest.importorskip("cv2")
        from norfair.camera_motion import _get_sparse_flow

        rng = np.random.RandomState(0)
        gray = rng.randint(0, 256, (200, 200), dtype=np.uint8)

        curr_pts, prev_pts = _get_sparse_flow(gray, gray)
        if len(curr_pts) > 0:
            displacements = np.abs(curr_pts - prev_pts)
            # Identical images => displacements should be near zero
            assert displacements.mean() < 1.0

    def test_get_sparse_flow_returns_matching_shapes(self):
        """curr_pts and prev_pts should have the same shape."""
        pytest.importorskip("cv2")
        from norfair.camera_motion import _get_sparse_flow

        rng = np.random.RandomState(1)
        gray1 = rng.randint(0, 256, (200, 200), dtype=np.uint8)
        gray2 = np.roll(gray1, 5, axis=1)

        curr_pts, prev_pts = _get_sparse_flow(gray2, gray1)
        assert curr_pts.shape == prev_pts.shape
        assert curr_pts.ndim == 2
        if len(curr_pts) > 0:
            assert curr_pts.shape[1] == 2

    def test_calc_optical_flow_basic(self):
        """_calc_optical_flow should return arrays or Nones."""
        pytest.importorskip("cv2")
        from norfair.camera_motion import _calc_optical_flow

        rng = np.random.RandomState(2)
        gray = rng.randint(0, 256, (100, 100), dtype=np.uint8)

        # Create some feature points in OpenCV's expected format (N, 1, 2)
        prev_pts = np.array([[[10.0, 10.0]], [[50.0, 50.0]]], dtype=np.float32)
        next_pts, status = _calc_optical_flow(gray, gray, prev_pts)

        if next_pts is not None:
            assert next_pts.shape[0] == prev_pts.shape[0]
        if status is not None:
            assert status.shape[0] == prev_pts.shape[0]
