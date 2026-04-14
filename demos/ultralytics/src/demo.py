"""Object detection and tracking using Ultralytics YOLO + Norfair."""

from __future__ import annotations

import argparse

import numpy as np
from ultralytics import YOLO

import norfair
from norfair import Detection, Tracker, Video

DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30


def detections_to_norfair(
    results,
    track_points: str = "bbox",
) -> list[Detection]:
    """Convert Ultralytics results to Norfair Detection objects."""
    detections: list[Detection] = []
    boxes = results[0].boxes

    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        cls = int(boxes.cls[i])

        if track_points == "centroid":
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            centroid = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
            detections.append(
                Detection(points=centroid, scores=np.array([conf]), label=cls)
            )
        elif track_points == "bbox":
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bbox = np.array([[x1, y1], [x2, y2]])
            detections.append(
                Detection(points=bbox, scores=np.array([conf, conf]), label=cls)
            )

    return detections


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument("files", type=str, nargs="+", help="Video files to process")
    parser.add_argument(
        "--model", type=str, default="yolo11n.pt", help="Ultralytics model name"
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=0.25, help="Confidence threshold"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Inference device: cpu, cuda, mps"
    )
    parser.add_argument(
        "--track-points",
        type=str,
        default="bbox",
        choices=["centroid", "bbox"],
        help="Track by 'centroid' or 'bbox'",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="Filter by class ID, e.g. --classes 0 2 3",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    model = YOLO(args.model)

    for input_path in args.files:
        video = Video(input_path=input_path)

        distance_function = "iou" if args.track_points == "bbox" else "euclidean"
        distance_threshold = (
            DISTANCE_THRESHOLD_BBOX
            if args.track_points == "bbox"
            else DISTANCE_THRESHOLD_CENTROID
        )

        tracker = Tracker(
            distance_function=distance_function,
            distance_threshold=distance_threshold,
        )

        for frame in video:
            results = model(
                frame,
                conf=args.conf_threshold,
                classes=args.classes,
                device=args.device,
                verbose=False,
            )
            detections = detections_to_norfair(results, track_points=args.track_points)
            tracked_objects = tracker.update(detections=detections)

            if args.track_points == "centroid":
                norfair.draw_points(frame, detections)
                norfair.draw_points(frame, tracked_objects)
            else:
                norfair.draw_boxes(frame, detections)
                norfair.draw_boxes(frame, tracked_objects)

            video.write(frame)


if __name__ == "__main__":
    run(make_parser().parse_args())
