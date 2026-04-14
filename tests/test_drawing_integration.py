"""Integration tests for norfair.drawing functions.

These tests exercise the public drawing API with real numpy arrays and
(when available) OpenCV, verifying that:
  - drawing functions produce visible output on a black canvas,
  - edge cases (NaN/Inf coordinates, empty/None drawables) are handled
    gracefully without crashes,
  - Paths, AbsolutePaths, FixedCamera, and draw_absolute_grid work
    with mock tracked objects and coordinate transforms.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from norfair import Detection
from norfair.drawing import (
    AbsolutePaths,
    Paths,
    draw_absolute_grid,
    draw_boxes,
    draw_points,
)
from norfair.drawing.drawer import Drawable
from norfair.drawing.fixed_camera import FixedCamera

# Visible color constant used in tests where the default palette color
# would be black (e.g. Detection objects have id=None, so Palette returns
# Color.black which is invisible on a black frame).
_GREEN = (0, 255, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _black_frame(size=200):
    """Return a size x size 3-channel black frame."""
    return np.zeros((size, size, 3), dtype=np.uint8)


def _frame_has_nonzero(frame):
    """Return True if the frame contains any non-zero pixels."""
    return frame.any()


class MockTrackedObject:
    """Minimal stand-in for TrackedObject, satisfying the drawing API.

    The drawing code accesses: .estimate, .id, .label, .scores,
    .live_points, .abs_to_rel, and .get_estimate().
    """

    def __init__(
        self,
        estimate=None,
        obj_id=1,
        label=None,
        live_points=None,
        scores=None,
        abs_to_rel=None,
    ):
        if estimate is None:
            estimate = np.array([[30, 30], [70, 70]])
        self.estimate = np.asarray(estimate, dtype=float)
        self.id = obj_id
        self.label = label
        self.scores = scores
        self.live_points = (
            live_points
            if live_points is not None
            else np.ones(len(self.estimate), dtype=bool)
        )
        self.abs_to_rel = abs_to_rel

    def get_estimate(self, absolute=False):
        return self.estimate


class MockCoordTransform:
    """Minimal coordinate transformation (identity)."""

    def abs_to_rel(self, points):
        return points

    def rel_to_abs(self, points):
        return points


# ---------------------------------------------------------------------------
# draw_boxes tests
# ---------------------------------------------------------------------------

class TestDrawBoxes:
    def test_basic_detection(self):
        """Detection drawn with an explicit visible color and thickness."""
        frame = _black_frame()
        det = Detection(points=np.array([[10, 10], [50, 50]]))
        result = draw_boxes(frame, [det], color=_GREEN, thickness=2)
        assert _frame_has_nonzero(result)

    def test_multiple_detections(self):
        frame = _black_frame()
        det1 = Detection(points=np.array([[10, 10], [50, 50]]))
        det2 = Detection(points=np.array([[60, 60], [90, 90]]))
        result = draw_boxes(frame, [det1, det2], color=_GREEN, thickness=2)
        assert _frame_has_nonzero(result)

    def test_with_color_tuple(self):
        frame = _black_frame()
        det = Detection(points=np.array([[10, 10], [50, 50]]))
        result = draw_boxes(frame, [det], color=(0, 255, 0), thickness=2)
        assert _frame_has_nonzero(result)

    def test_with_labels_and_ids(self):
        frame = _black_frame()
        det = Detection(points=np.array([[10, 10], [80, 80]]), label="cat")
        result = draw_boxes(
            frame, [det], color=_GREEN, thickness=2,
            draw_labels=True, draw_ids=True,
        )
        assert _frame_has_nonzero(result)

    def test_with_scores(self):
        frame = _black_frame()
        det = Detection(
            points=np.array([[10, 10], [80, 80]]),
            scores=np.array([0.9, 0.8]),
        )
        result = draw_boxes(
            frame, [det], color=_GREEN, thickness=2, draw_scores=True,
        )
        assert _frame_has_nonzero(result)

    def test_draw_box_false_still_draws_text(self):
        frame = _black_frame()
        det = Detection(points=np.array([[10, 10], [80, 80]]), label="dog")
        result = draw_boxes(
            frame, [det], color=_GREEN,
            draw_box=False, draw_labels=True,
        )
        # Even with draw_box=False, text may be drawn if label present.
        # Just ensure no crash.
        assert result is not None

    def test_with_explicit_thickness(self):
        frame = _black_frame()
        det = Detection(points=np.array([[10, 10], [50, 50]]))
        result = draw_boxes(frame, [det], color=_GREEN, thickness=3)
        assert _frame_has_nonzero(result)

    def test_returns_same_array(self):
        """draw_boxes modifies the frame in place and returns it."""
        frame = _black_frame()
        det = Detection(points=np.array([[10, 10], [50, 50]]))
        result = draw_boxes(frame, [det], color=_GREEN, thickness=2)
        assert result is frame

    def test_auto_thickness_on_large_frame(self):
        """On a large frame the auto-computed thickness is non-zero."""
        frame = _black_frame(size=800)
        det = Detection(points=np.array([[100, 100], [400, 400]]))
        result = draw_boxes(frame, [det], color=_GREEN)
        assert _frame_has_nonzero(result)


# ---------------------------------------------------------------------------
# draw_points tests
# ---------------------------------------------------------------------------

class TestDrawPoints:
    def test_basic_detection(self):
        frame = _black_frame()
        det = Detection(points=np.array([[10, 10], [50, 50]]))
        result = draw_points(frame, [det], color=_GREEN)
        assert _frame_has_nonzero(result)

    def test_multiple_detections(self):
        frame = _black_frame()
        det1 = Detection(points=np.array([[20, 20], [40, 40]]))
        det2 = Detection(points=np.array([[60, 60], [80, 80]]))
        result = draw_points(frame, [det1, det2], color=_GREEN)
        assert _frame_has_nonzero(result)

    def test_with_color_tuple(self):
        frame = _black_frame()
        det = Detection(points=np.array([[30, 30], [70, 70]]))
        result = draw_points(frame, [det], color=(255, 0, 0))
        assert _frame_has_nonzero(result)

    def test_with_custom_radius(self):
        frame = _black_frame()
        det = Detection(points=np.array([[50, 50], [100, 100]]))
        result = draw_points(frame, [det], color=_GREEN, radius=5)
        assert _frame_has_nonzero(result)

    def test_draw_points_false(self):
        """When draw_points=False only text is drawn, circles are omitted."""
        frame = _black_frame()
        det = Detection(points=np.array([[30, 30], [70, 70]]), label="test")
        # Should not crash even when circles are suppressed
        result = draw_points(
            frame, [det], color=_GREEN,
            draw_points=False, draw_labels=True,
        )
        assert result is not None

    def test_single_point_detection(self):
        frame = _black_frame()
        det = Detection(points=np.array([[100, 100]]))
        result = draw_points(frame, [det], color=_GREEN)
        assert _frame_has_nonzero(result)

    def test_with_text_color(self):
        frame = _black_frame()
        det = Detection(points=np.array([[50, 50], [100, 100]]), label="cat")
        result = draw_points(
            frame, [det], color=_GREEN,
            text_color=(0, 0, 255), draw_labels=True,
        )
        assert _frame_has_nonzero(result)


# ---------------------------------------------------------------------------
# NaN / Inf coordinate safety
# ---------------------------------------------------------------------------

class TestNanInfSafety:
    """Non-finite coordinates must not crash; the frame should stay unchanged."""

    def test_draw_boxes_nan(self):
        frame = _black_frame()
        original = frame.copy()
        det = Detection(points=np.array([[np.nan, np.nan], [50, 50]]))
        result = draw_boxes(frame, [det], color=_GREEN, thickness=2)
        np.testing.assert_array_equal(result, original)

    def test_draw_boxes_inf(self):
        frame = _black_frame()
        original = frame.copy()
        det = Detection(points=np.array([[np.inf, 10], [50, 50]]))
        result = draw_boxes(frame, [det], color=_GREEN, thickness=2)
        np.testing.assert_array_equal(result, original)

    def test_draw_boxes_negative_inf(self):
        frame = _black_frame()
        original = frame.copy()
        det = Detection(points=np.array([[-np.inf, 10], [50, 50]]))
        result = draw_boxes(frame, [det], color=_GREEN, thickness=2)
        np.testing.assert_array_equal(result, original)

    def test_draw_points_nan(self):
        frame = _black_frame()
        original = frame.copy()
        det = Detection(points=np.array([[np.nan, np.nan], [np.nan, np.nan]]))
        result = draw_points(frame, [det], color=_GREEN)
        np.testing.assert_array_equal(result, original)

    def test_draw_points_inf(self):
        frame = _black_frame()
        original = frame.copy()
        det = Detection(points=np.array([[np.inf, np.inf], [np.inf, np.inf]]))
        result = draw_points(frame, [det], color=_GREEN)
        np.testing.assert_array_equal(result, original)

    def test_draw_points_mixed_nan_valid(self):
        """One valid point, one NaN -- valid point should still draw."""
        frame = _black_frame()
        det = Detection(points=np.array([[50, 50], [np.nan, np.nan]]))
        result = draw_points(frame, [det], color=_GREEN)
        # The valid point should produce pixels
        assert _frame_has_nonzero(result)

    def test_draw_boxes_all_nan_detection(self):
        """A fully NaN detection should leave the frame untouched."""
        frame = _black_frame()
        original = frame.copy()
        det = Detection(points=np.array([[np.nan, np.nan], [np.nan, np.nan]]))
        result = draw_boxes(frame, [det], color=_GREEN, thickness=2)
        np.testing.assert_array_equal(result, original)


# ---------------------------------------------------------------------------
# Empty / None drawables
# ---------------------------------------------------------------------------

class TestEmptyNoneDrawables:
    def test_draw_boxes_empty_list(self):
        frame = _black_frame()
        original = frame.copy()
        result = draw_boxes(frame, [])
        np.testing.assert_array_equal(result, original)

    def test_draw_boxes_none(self):
        frame = _black_frame()
        original = frame.copy()
        result = draw_boxes(frame, None)
        np.testing.assert_array_equal(result, original)

    def test_draw_points_empty_list(self):
        frame = _black_frame()
        original = frame.copy()
        result = draw_points(frame, [])
        np.testing.assert_array_equal(result, original)

    def test_draw_points_none(self):
        frame = _black_frame()
        original = frame.copy()
        result = draw_points(frame, None)
        np.testing.assert_array_equal(result, original)

    def test_draw_boxes_default_drawables(self):
        """Calling with no drawables argument at all uses the default None."""
        frame = _black_frame()
        original = frame.copy()
        result = draw_boxes(frame)
        np.testing.assert_array_equal(result, original)

    def test_draw_points_default_drawables(self):
        frame = _black_frame()
        original = frame.copy()
        result = draw_points(frame)
        np.testing.assert_array_equal(result, original)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

class TestPaths:
    def test_draw_produces_output(self):
        frame = _black_frame()
        obj = MockTrackedObject(estimate=np.array([[50, 50], [80, 80]]), obj_id=1)
        path_drawer = Paths()
        result = path_drawer.draw(frame, [obj])
        # Palette picks a non-black color for id=1, so pixels should appear
        assert _frame_has_nonzero(result)

    def test_draw_multiple_frames(self):
        """Drawing across multiple frames accumulates a path."""
        path_drawer = Paths(attenuation=0.0)  # no fading
        for _ in range(5):
            frame = _black_frame()
            obj = MockTrackedObject(
                estimate=np.array([[50, 50], [80, 80]]), obj_id=1
            )
            result = path_drawer.draw(frame, [obj])
        assert _frame_has_nonzero(result)

    def test_draw_with_custom_color(self):
        frame = _black_frame()
        obj = MockTrackedObject(estimate=np.array([[40, 40], [60, 60]]), obj_id=2)
        path_drawer = Paths(color=(0, 0, 255), radius=3, thickness=2)
        result = path_drawer.draw(frame, [obj])
        assert _frame_has_nonzero(result)

    def test_draw_empty_objects(self):
        frame = _black_frame()
        path_drawer = Paths()
        result = path_drawer.draw(frame, [])
        # With no objects, result should still be valid (blended black frames)
        assert result is not None

    def test_draw_with_nan_estimate(self):
        """NaN estimates should not crash the path drawer."""
        frame = _black_frame()
        obj = MockTrackedObject(
            estimate=np.array([[np.nan, np.nan], [np.nan, np.nan]]), obj_id=3
        )
        path_drawer = Paths()
        result = path_drawer.draw(frame, [obj])
        assert result is not None

    def test_draw_with_attenuation(self):
        """Non-zero attenuation should still produce visible output."""
        path_drawer = Paths(attenuation=0.5)
        frame = _black_frame()
        obj = MockTrackedObject(estimate=np.array([[50, 50], [80, 80]]), obj_id=1)
        result = path_drawer.draw(frame, [obj])
        assert _frame_has_nonzero(result)


# ---------------------------------------------------------------------------
# AbsolutePaths
# ---------------------------------------------------------------------------

class TestAbsolutePaths:
    def test_draw_without_transform(self):
        frame = _black_frame()
        obj = MockTrackedObject(estimate=np.array([[50, 50], [80, 80]]), obj_id=1)
        abs_paths = AbsolutePaths()
        result = abs_paths.draw(frame, [obj])
        assert _frame_has_nonzero(result)

    def test_draw_with_transform(self):
        frame = _black_frame()
        obj = MockTrackedObject(estimate=np.array([[50, 50], [80, 80]]), obj_id=1)
        transform = MockCoordTransform()
        abs_paths = AbsolutePaths()
        result = abs_paths.draw(frame, [obj], coord_transform=transform)
        assert _frame_has_nonzero(result)

    def test_draw_multiple_frames_accumulates(self):
        abs_paths = AbsolutePaths(max_history=5)
        transform = MockCoordTransform()
        for i in range(5):
            frame = _black_frame()
            obj = MockTrackedObject(
                estimate=np.array(
                    [[30 + i * 5, 30 + i * 5], [60 + i * 5, 60 + i * 5]]
                ),
                obj_id=1,
            )
            result = abs_paths.draw(frame, [obj], coord_transform=transform)
        assert _frame_has_nonzero(result)

    def test_draw_with_custom_settings(self):
        frame = _black_frame()
        obj = MockTrackedObject(estimate=np.array([[40, 40], [90, 90]]), obj_id=2)
        abs_paths = AbsolutePaths(color=(255, 0, 0), radius=4, thickness=2)
        result = abs_paths.draw(frame, [obj])
        assert _frame_has_nonzero(result)

    def test_draw_empty_objects(self):
        frame = _black_frame()
        abs_paths = AbsolutePaths()
        result = abs_paths.draw(frame, [])
        assert result is not None

    def test_dead_points_object_skipped(self):
        """Objects with all dead live_points should be skipped."""
        frame = _black_frame()
        original = frame.copy()
        obj = MockTrackedObject(
            estimate=np.array([[50, 50], [80, 80]]),
            obj_id=1,
            live_points=np.array([False, False]),
        )
        abs_paths = AbsolutePaths()
        result = abs_paths.draw(frame, [obj])
        # No live points => nothing drawn, frame should be unchanged
        np.testing.assert_array_equal(result, original)

    def test_cleanup_dead_object_ids(self):
        """Past points for objects no longer tracked should be cleaned up."""
        abs_paths = AbsolutePaths()
        obj1 = MockTrackedObject(estimate=np.array([[50, 50], [80, 80]]), obj_id=1)
        obj2 = MockTrackedObject(estimate=np.array([[20, 20], [40, 40]]), obj_id=2)

        # Frame 1: both objects
        frame = _black_frame()
        abs_paths.draw(frame, [obj1, obj2])
        assert 1 in abs_paths.past_points
        assert 2 in abs_paths.past_points

        # Frame 2: only obj1 present
        frame = _black_frame()
        abs_paths.draw(frame, [obj1])
        assert 1 in abs_paths.past_points
        assert 2 not in abs_paths.past_points


# ---------------------------------------------------------------------------
# draw_absolute_grid
# ---------------------------------------------------------------------------

class TestDrawAbsoluteGrid:
    def test_no_transform(self):
        """Grid with a visible color and no coord transform."""
        frame = _black_frame()
        draw_absolute_grid(
            frame, coord_transformations=None, grid_size=10, color=_GREEN,
        )
        assert _frame_has_nonzero(frame)

    def test_with_identity_transform(self):
        frame = _black_frame()
        transform = MockCoordTransform()
        draw_absolute_grid(
            frame, coord_transformations=transform, grid_size=10, color=_GREEN,
        )
        assert _frame_has_nonzero(frame)

    def test_custom_color(self):
        frame = _black_frame()
        draw_absolute_grid(
            frame,
            coord_transformations=None,
            grid_size=10,
            color=(0, 255, 0),
        )
        assert _frame_has_nonzero(frame)

    def test_polar_mode(self):
        frame = _black_frame()
        draw_absolute_grid(
            frame, coord_transformations=None, grid_size=10,
            polar=True, color=_GREEN,
        )
        assert _frame_has_nonzero(frame)

    def test_custom_radius_and_thickness(self):
        frame = _black_frame()
        draw_absolute_grid(
            frame,
            coord_transformations=None,
            grid_size=10,
            radius=5,
            thickness=2,
            color=_GREEN,
        )
        assert _frame_has_nonzero(frame)

    def test_large_grid_size(self):
        frame = _black_frame()
        draw_absolute_grid(
            frame, coord_transformations=None, grid_size=50, color=_GREEN,
        )
        assert _frame_has_nonzero(frame)


# ---------------------------------------------------------------------------
# FixedCamera
# ---------------------------------------------------------------------------

class TestFixedCamera:
    def test_basic_adjust_frame(self):
        """FixedCamera should produce a larger canvas with the frame embedded."""
        from norfair.camera_motion import TranslationTransformation

        frame = _black_frame()
        # Draw something on the frame so we can verify it ends up in the output
        cv2.rectangle(frame, (10, 10), (50, 50), (255, 255, 255), 2)

        transform = TranslationTransformation(movement_vector=np.array([0, 0]))
        camera = FixedCamera(scale=2)
        result = camera.adjust_frame(frame, transform)

        # Output should be larger than input
        assert result.shape[0] == 400
        assert result.shape[1] == 400
        # The white rectangle should be present somewhere in the output
        assert _frame_has_nonzero(result)

    def test_with_translation(self):
        from norfair.camera_motion import TranslationTransformation

        frame = _black_frame()
        cv2.rectangle(frame, (10, 10), (50, 50), (0, 0, 255), -1)

        transform = TranslationTransformation(movement_vector=np.array([5, 5]))
        camera = FixedCamera(scale=3)
        result = camera.adjust_frame(frame, transform)

        assert result.shape[0] == 600
        assert result.shape[1] == 600
        assert _frame_has_nonzero(result)

    def test_multiple_frames(self):
        from norfair.camera_motion import TranslationTransformation

        camera = FixedCamera(scale=2, attenuation=0.05)
        for i in range(5):
            frame = _black_frame()
            cv2.circle(frame, (100, 100), 10, (0, 255, 0), -1)
            transform = TranslationTransformation(
                movement_vector=np.array([i * 2, i * 2])
            )
            result = camera.adjust_frame(frame, transform)

        assert _frame_has_nonzero(result)

    def test_custom_scale(self):
        from norfair.camera_motion import TranslationTransformation

        frame = _black_frame()
        cv2.circle(frame, (50, 50), 5, (128, 128, 128), -1)

        transform = TranslationTransformation(movement_vector=np.array([0, 0]))
        camera = FixedCamera(scale=1.5)
        result = camera.adjust_frame(frame, transform)

        assert result.shape[0] == 300
        assert result.shape[1] == 300
        assert _frame_has_nonzero(result)

    def test_zero_translation(self):
        """A zero-vector translation should center the frame in the canvas."""
        from norfair.camera_motion import TranslationTransformation

        frame = _black_frame()
        cv2.circle(frame, (100, 100), 20, (255, 128, 0), -1)

        transform = TranslationTransformation(movement_vector=np.array([0, 0]))
        camera = FixedCamera(scale=2)
        result = camera.adjust_frame(frame, transform)
        assert _frame_has_nonzero(result)


# ---------------------------------------------------------------------------
# Drawable wrapper
# ---------------------------------------------------------------------------

class TestDrawable:
    def test_from_detection(self):
        det = Detection(points=np.array([[10, 20], [30, 40]]), label="car")
        d = Drawable(det)
        assert d.label == "car"
        assert d.id is None
        np.testing.assert_array_equal(d.points, det.points)
        assert d.live_points.all()

    def test_from_detection_with_scores(self):
        det = Detection(
            points=np.array([[10, 20], [30, 40]]),
            scores=np.array([0.9, 0.8]),
        )
        d = Drawable(det)
        np.testing.assert_array_equal(d.scores, np.array([0.9, 0.8]))

    def test_from_explicit_fields(self):
        pts = np.array([[5, 5], [15, 15]])
        lp = np.array([True, False])
        d = Drawable(
            points=pts,
            id=42,
            label="person",
            live_points=lp,
        )
        assert d.id == 42
        assert d.label == "person"
        np.testing.assert_array_equal(d.points, pts)
        np.testing.assert_array_equal(d.live_points, lp)

    def test_invalid_obj_type_raises(self):
        with pytest.raises(ValueError, match="Expecting a Detection"):
            Drawable("not a detection")

    def test_from_none_with_explicit_fields(self):
        pts = np.array([[1, 2]])
        d = Drawable(obj=None, points=pts, id=99)
        assert d.id == 99
        np.testing.assert_array_equal(d.points, pts)


# ---------------------------------------------------------------------------
# draw_boxes / draw_points with Drawable wrapper
# ---------------------------------------------------------------------------

class TestDrawableWithDrawFunctions:
    def test_draw_boxes_accepts_drawable(self):
        frame = _black_frame()
        d = Drawable(
            points=np.array([[10, 10], [60, 60]]),
            id=1,
            live_points=np.array([True, True]),
        )
        result = draw_boxes(frame, [d], color=_GREEN, thickness=2)
        assert _frame_has_nonzero(result)

    def test_draw_points_accepts_drawable(self):
        frame = _black_frame()
        d = Drawable(
            points=np.array([[30, 30], [70, 70]]),
            id=2,
            live_points=np.array([True, True]),
        )
        result = draw_points(frame, [d], color=_GREEN)
        assert _frame_has_nonzero(result)


# ---------------------------------------------------------------------------
# Color strategy variants for draw_boxes
# ---------------------------------------------------------------------------

class TestColorStrategies:
    def test_by_id_with_real_id(self):
        """Drawable with a real integer id gets a non-black palette color."""
        frame = _black_frame()
        d = Drawable(
            points=np.array([[10, 10], [50, 50]]),
            id=7,
            live_points=np.array([True, True]),
        )
        result = draw_boxes(frame, [d], color="by_id", thickness=2)
        assert _frame_has_nonzero(result)

    def test_by_label(self):
        frame = _black_frame()
        det = Detection(points=np.array([[10, 10], [50, 50]]), label="truck")
        result = draw_boxes(frame, [det], color="by_label", thickness=2)
        assert _frame_has_nonzero(result)

    def test_random(self):
        frame = _black_frame()
        det = Detection(points=np.array([[10, 10], [50, 50]]))
        # "random" uses np.random.rand() as the hash key; very unlikely
        # to be black.
        result = draw_boxes(frame, [det], color="random", thickness=2)
        assert _frame_has_nonzero(result)

    def test_hex_color(self):
        frame = _black_frame()
        det = Detection(points=np.array([[10, 10], [50, 50]]))
        result = draw_boxes(frame, [det], color="#FF0000", thickness=2)
        assert _frame_has_nonzero(result)

    def test_explicit_bgr_tuple(self):
        frame = _black_frame()
        det = Detection(points=np.array([[10, 10], [50, 50]]))
        result = draw_boxes(frame, [det], color=(255, 128, 0), thickness=2)
        assert _frame_has_nonzero(result)
