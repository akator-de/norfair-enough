import argparse

import numpy as np
from draw import draw
from ultralytics import YOLO

from norfair import AbsolutePaths, Detection, Tracker, Video
from norfair.camera_motion import HomographyTransformationGetter, MotionEstimator
from norfair.distances import create_normalized_mean_euclidean_distance

DISTANCE_THRESHOLD_CENTROID: float = 0.08


def ultralytics_to_norfair(results, track_points: str = "bbox") -> list[Detection]:
    """Convert Ultralytics detection results to Norfair Detections."""
    norfair_detections: list[Detection] = []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return norfair_detections

    for box in boxes:
        conf = float(box.conf[0])
        if track_points == "centroid":
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            points = np.array([[cx, cy], [cx, cy]])
            scores = np.array([conf, conf])
        else:  # bbox
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            points = np.array([[x1, y1], [x2, y2]])
            scores = np.array([conf, conf])
        norfair_detections.append(Detection(points=points, scores=scores))

    return norfair_detections


def inference(
    input_video: str,
    model_path: str,
    track_points: str,
    conf_threshold: float,
    classes: list[int] | None,
):
    model = YOLO(model_path)
    video = Video(input_path=input_video)

    transformations_getter = HomographyTransformationGetter()
    motion_estimator = MotionEstimator(
        max_points=500, min_distance=7, transformations_getter=transformations_getter
    )

    distance_function = create_normalized_mean_euclidean_distance(
        video.input_height, video.input_width
    )
    distance_threshold = DISTANCE_THRESHOLD_CENTROID

    tracker = Tracker(
        distance_function=distance_function,
        distance_threshold=distance_threshold,
    )

    paths_drawer = AbsolutePaths(max_history=40, thickness=2)
    fix_paths = True

    for frame in video:
        results = model(frame, conf=conf_threshold, iou=0.45, classes=classes)
        detections = ultralytics_to_norfair(results, track_points=track_points)

        mask = np.ones(frame.shape[:2], frame.dtype)
        coord_transformations = motion_estimator.update(frame, mask)

        tracked_objects = tracker.update(
            detections=detections, coord_transformations=coord_transformations
        )

        frame = draw(
            paths_drawer,
            track_points,
            frame,
            detections,
            tracked_objects,
            coord_transformations,
            fix_paths,
        )
        video.write(frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument("files", type=str, help="Video files to process")
    parser.add_argument(
        "--detector-path",
        type=str,
        default="yolo11n.pt",
        help="Ultralytics YOLO model path",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="YOLO object confidence threshold",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="Filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--track-points",
        type=str,
        default="bbox",
        help="Track points: 'centroid' or 'bbox'",
    )
    args = parser.parse_args()

    inference(
        args.files,
        args.detector_path,
        args.track_points,
        args.conf_threshold,
        args.classes,
    )
