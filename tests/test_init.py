"""Test that public API exports are accessible from the top-level package."""

import pytest


def test_tracker_exports():
    """Test that tracker classes are exported."""
    from norfair import Detection, Tracker, TrackedObject

    assert Detection is not None
    assert Tracker is not None
    assert TrackedObject is not None


def test_distance_exports():
    """Test that distance classes and functions are exported."""
    from norfair import (
        Distance,
        ScalarDistance,
        ScipyDistance,
        VectorizedDistance,
        create_keypoints_voting_distance,
        create_normalized_mean_euclidean_distance,
        frobenius,
        get_distance_by_name,
        iou,
        iou_opt,
        mean_euclidean,
        mean_manhattan,
    )

    assert Distance is not None
    assert ScalarDistance is not None
    assert ScipyDistance is not None
    assert VectorizedDistance is not None
    assert create_keypoints_voting_distance is not None
    assert create_normalized_mean_euclidean_distance is not None
    assert frobenius is not None
    assert get_distance_by_name is not None
    assert iou is not None
    assert iou_opt is not None
    assert mean_euclidean is not None
    assert mean_manhattan is not None


def test_filter_exports():
    """Test that filter factory classes are exported."""
    from norfair import (
        FilterFactory,
        FilterPyKalmanFilterFactory,
        NoFilterFactory,
        OptimizedKalmanFilterFactory,
    )

    assert FilterFactory is not None
    assert FilterPyKalmanFilterFactory is not None
    assert NoFilterFactory is not None
    assert OptimizedKalmanFilterFactory is not None


def test_drawing_exports():
    """Test that drawing classes and functions are exported."""
    from norfair import (
        AbsolutePaths,
        Color,
        ColorLike,
        ColorType,
        Drawable,
        FixedCamera,
        Palette,
        Paths,
        draw_absolute_grid,
        draw_boxes,
        draw_points,
        draw_tracked_boxes,
        draw_tracked_objects,
    )

    assert AbsolutePaths is not None
    assert Color is not None
    assert ColorLike is not None
    assert ColorType is not None
    assert Drawable is not None
    assert FixedCamera is not None
    assert Palette is not None
    assert Paths is not None
    assert draw_absolute_grid is not None
    assert draw_boxes is not None
    assert draw_points is not None
    assert draw_tracked_boxes is not None
    assert draw_tracked_objects is not None


def test_camera_motion_exports():
    """Test that camera motion classes are exported."""
    from norfair import (
        CoordinatesTransformation,
        HomographyTransformation,
        HomographyTransformationGetter,
        MotionEstimator,
        TransformationGetter,
        TranslationTransformation,
        TranslationTransformationGetter,
    )

    assert CoordinatesTransformation is not None
    assert HomographyTransformation is not None
    assert HomographyTransformationGetter is not None
    assert MotionEstimator is not None
    assert TransformationGetter is not None
    assert TranslationTransformation is not None
    assert TranslationTransformationGetter is not None


def test_metrics_exports():
    """Test that metrics classes are exported."""
    from norfair import (
        Accumulators,
        DetectionFileParser,
        InformationFile,
        PredictionsTextFile,
    )

    assert Accumulators is not None
    assert DetectionFileParser is not None
    assert InformationFile is not None
    assert PredictionsTextFile is not None


def test_utils_exports():
    """Test that utility functions are exported."""
    from norfair import get_cutout, print_objects_as_table

    assert get_cutout is not None
    assert print_objects_as_table is not None


def test_video_exports():
    """Test that Video class is exported."""
    from norfair import Video

    assert Video is not None


def test_all_exports_in_all_list():
    """Test that __all__ contains all expected exports."""
    import norfair

    # Check that __all__ is defined
    assert hasattr(norfair, "__all__")
    all_exports = norfair.__all__

    # Key exports that should be in __all__
    expected_exports = [
        # tracker
        "Detection",
        "Tracker",
        "TrackedObject",
        # distances
        "Distance",
        "ScalarDistance",
        "VectorizedDistance",
        "ScipyDistance",
        # filters
        "FilterFactory",
        "OptimizedKalmanFilterFactory",
        "NoFilterFactory",
        # drawing
        "ColorLike",
        "Drawable",
        "draw_tracked_objects",
        # camera_motion
        "MotionEstimator",
        "CoordinatesTransformation",
        # metrics
        "Accumulators",
        # video
        "Video",
    ]

    for export in expected_exports:
        assert export in all_exports, f"{export} not in __all__"


def test_version_available():
    """Test that __version__ is available."""
    import norfair

    assert hasattr(norfair, "__version__")
    assert isinstance(norfair.__version__, str)
    assert len(norfair.__version__) > 0


def test_imports_dont_fail():
    """Test that importing norfair doesn't raise any errors."""
    import norfair

    assert norfair is not None


def test_can_create_tracker_from_top_level():
    """Test that we can create a Tracker using top-level imports."""
    from norfair import Detection, Tracker

    tracker = Tracker(distance_function="euclidean", distance_threshold=10)
    assert tracker is not None

    # Create a detection and update tracker
    import numpy as np

    det = Detection(points=np.array([[1, 2]]))
    tracked = tracker.update([det])
    assert isinstance(tracked, list)


def test_distance_function_from_name():
    """Test that we can get distance functions by name from top-level."""
    from norfair import get_distance_by_name

    distance = get_distance_by_name("euclidean")
    assert distance is not None


def test_video_context_manager_from_top_level():
    """Test that Video context manager works from top-level import."""
    from unittest.mock import patch

    from norfair import Video

    # We can't test actual video functionality without files, but we can
    # verify the import works and the class has the expected interface
    assert hasattr(Video, "__enter__")
    assert hasattr(Video, "__exit__")
    assert hasattr(Video, "close")