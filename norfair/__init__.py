"""
A customizable lightweight Python library for real-time multi-object tracking.

Examples
--------
>>> from norfair import Detection, Tracker, Video, draw_tracked_objects
>>> detector = MyDetector()  # Set up a detector
>>> video = Video(input_path="video.mp4")
>>> tracker = Tracker(distance_function="euclidean", distance_threshold=50)
>>> for frame in video:
>>>    detections = detector(frame)
>>>    norfair_detections = [Detection(points) for points in detections]
>>>    tracked_objects = tracker.update(detections=norfair_detections)
>>>    draw_tracked_objects(frame, tracked_objects)
>>>    video.write(frame)
"""

import importlib.metadata

from .camera_motion import (
    CoordinatesTransformation,
    HomographyTransformation,
    HomographyTransformationGetter,
    MotionEstimator,
    TransformationGetter,
    TranslationTransformation,
    TranslationTransformationGetter,
)
from .distances import (
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
from .drawing import (
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
from .filter import (
    FilterFactory,
    FilterPyKalmanFilterFactory,
    NoFilterFactory,
    OptimizedKalmanFilterFactory,
)
from .metrics import (
    Accumulators,
    DetectionFileParser,
    InformationFile,
    PredictionsTextFile,
)
from .tracker import Detection, TrackedObject, Tracker
from .utils import get_cutout, print_objects_as_table
from .video import Video

__version__ = importlib.metadata.version("norfair-enough")

__all__ = [
    # camera_motion
    "CoordinatesTransformation",
    "HomographyTransformation",
    "HomographyTransformationGetter",
    "MotionEstimator",
    "TransformationGetter",
    "TranslationTransformation",
    "TranslationTransformationGetter",
    # distances
    "Distance",
    "ScalarDistance",
    "ScipyDistance",
    "VectorizedDistance",
    "create_keypoints_voting_distance",
    "create_normalized_mean_euclidean_distance",
    "frobenius",
    "get_distance_by_name",
    "iou",
    "iou_opt",
    "mean_euclidean",
    "mean_manhattan",
    # drawing
    "AbsolutePaths",
    "Color",
    "ColorLike",
    "ColorType",
    "Drawable",
    "FixedCamera",
    "Palette",
    "Paths",
    "draw_absolute_grid",
    "draw_boxes",
    "draw_points",
    "draw_tracked_boxes",
    "draw_tracked_objects",
    # filter
    "FilterFactory",
    "FilterPyKalmanFilterFactory",
    "NoFilterFactory",
    "OptimizedKalmanFilterFactory",
    # metrics
    "Accumulators",
    "DetectionFileParser",
    "InformationFile",
    "PredictionsTextFile",
    # tracker
    "Detection",
    "Tracker",
    "TrackedObject",
    # utils
    "get_cutout",
    "print_objects_as_table",
    # video
    "Video",
]
