"""Track human pose keypoints using Ultralytics YOLO-Pose and Norfair."""

from __future__ import annotations

import argparse

import numpy as np
from ultralytics import YOLO

import norfair
from norfair import Detection, Tracker, Video
from norfair.distances import create_normalized_mean_euclidean_distance

DISTANCE_THRESHOLD: float = 0.3


def yolo_pose_to_detections(results) -> list[Detection]:
    """Convert Ultralytics pose results to Norfair Detections."""
    detections: list[Detection] = []
    if results[0].keypoints is None:
        return detections

    keypoints_xy = results[0].keypoints.xy.cpu().numpy()  # (N, 17, 2)
    conf = results[0].keypoints.conf
    keypoints_conf = (
        conf.cpu().numpy() if conf is not None else np.ones(keypoints_xy.shape[:2])
    )  # (N, 17)

    for points, scores in zip(keypoints_xy, keypoints_conf):
        detections.append(Detection(points=points, scores=scores))
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Track human pose keypoints in video.")
    parser.add_argument("files", type=str, nargs="+", help="Video files to process")
    parser.add_argument(
        "--model", type=str, default="yolo11n-pose.pt", help="YOLO-Pose model name"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Inference device: 'cpu', 'cuda', etc."
    )
    args = parser.parse_args()

    model = YOLO(args.model)

    for input_path in args.files:
        video = Video(input_path=input_path)

        distance_fn = create_normalized_mean_euclidean_distance(
            video.input_height, video.input_width
        )
        tracker = Tracker(
            distance_function=distance_fn,
            distance_threshold=DISTANCE_THRESHOLD,
        )

        for frame in video:
            results = model(frame, conf=args.conf_threshold, device=args.device)
            detections = yolo_pose_to_detections(results)
            tracked_objects = tracker.update(detections=detections)
            norfair.draw_points(frame, tracked_objects)
            video.write(frame)


if __name__ == "__main__":
    main()
