<p align="center">
  <img src="docs/img/banner.svg" alt="norfair-enough" width="100%">
</p>

[![CI](https://github.com/akator-de/norfair-enough/actions/workflows/ci.yml/badge.svg)](https://github.com/akator-de/norfair-enough/actions/workflows/ci.yml)
[![Lint](https://github.com/akator-de/norfair-enough/actions/workflows/lint.yml/badge.svg)](https://github.com/akator-de/norfair-enough/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/akator-de/norfair-enough/graph/badge.svg)](https://codecov.io/gh/akator-de/norfair-enough)
[![PyPI](https://img.shields.io/pypi/v/norfair-enough)](https://pypi.org/project/norfair-enough/)
[![Python](https://img.shields.io/pypi/pyversions/norfair-enough)](https://pypi.org/project/norfair-enough/)
[![Docs](https://img.shields.io/badge/docs-dev-blue)](https://akator-de.github.io/norfair-enough/)
[![License](https://img.shields.io/github/license/akator-de/norfair-enough)](https://github.com/akator-de/norfair-enough/blob/main/LICENSE)

**A maintained fork of [tryolabs/norfair](https://github.com/tryolabs/norfair)** — lightweight Python library for real-time multi-object tracking.

The upstream Norfair repository is no longer actively maintained. This fork keeps the library alive for production use. There is no claim to be the official successor — just a pragmatic continuation. New maintainers are welcome.

|                                           Tracking players with moving camera                                           |                                           Tracking 3D objects                                           |
| :---------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: |
| ![Tracking players in a soccer match](https://media.githubusercontent.com/media/akator-de/norfair-enough/main/docs/videos/soccer.webp) | ![Tracking objects in 3D](https://media.githubusercontent.com/media/akator-de/norfair-enough/main/docs/videos/3d.webp) |

## Installation

norfair-enough currently supports Python 3.10+.

For the minimal version, install as:

```bash
pip install norfair-enough
```

To install with optional dependencies:

```bash
pip install norfair-enough[video]    # Adds several video helper features running on OpenCV
pip install norfair-enough[metrics]  # Supports running MOT metrics evaluation
pip install norfair-enough[metrics,video]  # Everything included
```

> **Import note:** The Python import remains unchanged — use `from norfair import ...` as before.

If the needed dependencies are already present in the system, installing the minimal version is enough for enabling the extra features. This is particularly useful for embedded devices, where installing compiled dependencies can be difficult, but they can sometimes come preinstalled with the system.

## Features

- Any detector expressing its detections as a series of `(x, y)` coordinates can be used with Norfair Enough. This includes detectors performing tasks such as object or keypoint detection (see [examples](#examples--demos)).

- Modular. It can easily be inserted into complex video processing pipelines to add tracking to existing projects. At the same time, it is possible to build a video inference loop from scratch using just the library and a detector.

- Supports moving camera, re-identification with appearance embeddings, and n-dimensional object tracking (see [Advanced features](#advanced-features)).

- The library provides several predefined distance functions to compare tracked objects and detections. The distance functions can also be defined by the user, enabling the implementation of different tracking strategies.

- Fast. The only thing bounding inference speed will be the detection network feeding detections to the tracker.

## Documentation

[Getting started guide](https://akator-de.github.io/norfair-enough/dev/getting_started/).

[API reference](https://akator-de.github.io/norfair-enough/dev/reference/).

## Examples & demos

We provide several examples of how the library can be used to add tracking capabilities to different detectors, and also showcase more advanced features. All demos are available in the [`demos/`](https://github.com/akator-de/norfair-enough/tree/main/demos) directory of this repository (originally adapted from [tryolabs/norfair](https://github.com/tryolabs/norfair/tree/master/demos)).

> **Note:** The demo code was originally written in the [tryolabs/norfair](https://github.com/tryolabs/norfair) repository which is no longer actively maintained. Some demos may reference upstream resources that could become unavailable in the future. The demos remain compatible with norfair-enough — just replace `pip install norfair` with `pip install norfair-enough`.

> Some demos include Dockerfiles for reproducibility. If you have a GPU, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU passthrough. CPU-only usage is possible but may require dependency adjustments.

### Adding tracking to different detectors

Most tracking demos are showcased with vehicles and pedestrians, but the detectors are generally trained with many more classes from the [COCO dataset](https://cocodataset.org/).

1. [YOLOv7](https://github.com/akator-de/norfair-enough/tree/main/demos/yolov7): tracking object centroids or bounding boxes.
2. [YOLOv5](https://github.com/akator-de/norfair-enough/tree/main/demos/yolov5): tracking object centroids or bounding boxes.
3. [YOLOv4](https://github.com/akator-de/norfair-enough/tree/main/demos/yolov4): tracking object centroids.
4. [Detectron2](https://github.com/akator-de/norfair-enough/tree/main/demos/detectron2): tracking object centroids.
5. [AlphaPose](https://github.com/akator-de/norfair-enough/tree/main/demos/alphapose): tracking human keypoints (pose estimation) and inserting the tracker into a complex existing pipeline.
6. [OpenPose](https://github.com/akator-de/norfair-enough/tree/main/demos/openpose): tracking human keypoints.
7. [YOLOPv2](https://github.com/akator-de/norfair-enough/tree/main/demos/yolopv2): tracking with a model for traffic object detection, drivable road area segmentation, and lane line detection.
8. [YOLO-NAS](https://github.com/akator-de/norfair-enough/tree/main/demos/yolo_nas): tracking object centroids or bounding boxes.

### Advanced features

1. [Speed up pose estimation by extrapolating detections](https://github.com/akator-de/norfair-enough/tree/main/demos/openpose) using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
2. [Track both bounding boxes and human keypoints](https://github.com/akator-de/norfair-enough/tree/main/demos/keypoints_bounding_boxes) (multi-class), unifying the detections from a YOLO model and OpenPose.
3. [Re-identification (ReID)](https://github.com/akator-de/norfair-enough/tree/main/demos/reid) of tracked objects using appearance embeddings. This is a good starting point for scenarios with a lot of occlusion, in which the Kalman filter alone would struggle.
4. [Accurately track objects even if the camera is moving](https://github.com/akator-de/norfair-enough/tree/main/demos/camera_motion), by estimating camera motion potentially accounting for pan, tilt, rotation, movement in any direction, and zoom.
5. [Track points in 3D](https://github.com/akator-de/norfair-enough/tree/main/demos/3d_track), using [MediaPipe Objectron](https://google.github.io/mediapipe/solutions/objectron.html).
6. [Tracking of small objects](https://github.com/akator-de/norfair-enough/tree/main/demos/sahi), using [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi).

### Benchmarking and profiling

1. [Kalman filter and distance function profiling](https://github.com/akator-de/norfair-enough/tree/main/demos/profiling) using [TRT pose estimator](https://github.com/NVIDIA-AI-IOT/trt_pose).
2. Computation of [MOT17](https://motchallenge.net/data/MOT17/) scores using [motmetrics4norfair](https://github.com/akator-de/norfair-enough/tree/main/demos/motmetrics4norfair).

## How it works

The tracker works by estimating the future position of each point based on its past positions. It then tries to match these estimated positions with newly detected points provided by the detector. For this matching to occur, the tracker can rely on any distance function. There are some predefined distances already integrated in the library, and the users can also define their own custom distances. Therefore, each object tracker can be made as simple or as complex as needed.

As an example we use [Detectron2](https://github.com/facebookresearch/detectron2) to get the single point detections to use with this distance function. We just use the centroids of the bounding boxes it produces around cars as our detections, and get the following results.

![Tracking cars with Norfair Enough](https://media.githubusercontent.com/media/akator-de/norfair-enough/main/docs/videos/traffic.webp)

On the left you can see the points we get from Detectron2, and on the right how the tracker tracks them assigning a unique identifier through time. Even a straightforward distance function like this one can work when the tracking needed is simple.

The library also provides several useful tools for creating a video inference loop. Here is what the full code for creating the previous example looks like, including the code needed to set up Detectron2:

```python
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from norfair import Detection, Tracker, Video, draw_points

# Set up Detectron2 object detector
cfg = get_cfg()
cfg.merge_from_file("demos/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
detector = DefaultPredictor(cfg)

# Norfair
video = Video(input_path="video.mp4")
tracker = Tracker(distance_function="euclidean", distance_threshold=20)

for frame in video:
    detections = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detections = [Detection(p) for p in detections['instances'].pred_boxes.get_centers().cpu().numpy()]
    tracked_objects = tracker.update(detections=detections)
    draw_points(frame, drawables=tracked_objects)
    video.write(frame)
```

The video and drawing tools use OpenCV frames, so they are compatible with most Python video code available online. The point tracking is based on [SORT](https://arxiv.org/abs/1602.00763) generalized to detections consisting of a dynamically changing number of points per detection.

## Motivation

Modern object detectors are increasingly easy to use (e.g., Ultralytics YOLO), but adding robust multi-object tracking on top of them still requires stitching together detection, state estimation, and identity management. Norfair Enough was born to fill that gap: a modular tracking layer that works with any detector outputting `(x, y)` coordinates, and can be plugged into existing pipelines with minimal effort.

## Comparison to other trackers

Norfair Enough's contribution to Python's object tracker library repertoire is its ability to work with any object detector by being able to work with a variable number of points per detection, and the ability for the user to heavily customize the tracker by creating their own distance function.

If you are looking for a tracker, here are some other projects worth noting:

- [**ByteTrack**](https://github.com/ifzhang/ByteTrack) and [**BoT-SORT**](https://github.com/NirAharon/BoT-SORT) are high-performance MOT trackers that achieve strong results on MOT benchmarks. They are tightly coupled to specific detection architectures.
- [**Ultralytics built-in tracking**](https://docs.ultralytics.com/modes/track/) provides integrated tracking (ByteTrack, BoT-SORT) when using YOLO models. Convenient if you only use YOLO, but not detector-agnostic.
- [**SORT**](https://github.com/abewley/sort) and [**Deep SORT**](https://github.com/nwojke/deep_sort) use Kalman filters like Norfair Enough, but are hardcoded to bounding-box tracking with a fixed distance function. Both are released under the GPL.
- [**OC-SORT**](https://github.com/noahcao/OC_SORT) improves on SORT with observation-centric momentum, handling occlusion better. Like SORT, it is box-only.
- [**supervision**](https://github.com/roboflow/supervision) by Roboflow offers ByteTrack integration alongside annotation and dataset tools. Useful if you want a broader CV toolkit, but less customizable for tracking specifically.

Norfair Enough stands out by being **detector-agnostic**, supporting **any point geometry** (centroids, bounding boxes, keypoints, 3D points), and offering a **BSD-3 license** with no copyleft restrictions.

## Benchmarks

These benchmarks were produced using the [motmetrics4norfair](https://github.com/akator-de/norfair-enough/tree/main/demos/motmetrics4norfair) demo script. Our CI runs MOT metrics regression tests on every pull request to prevent tracking quality regressions.

[MOT17](https://motchallenge.net/data/MOT17/) and [MOT20](https://motchallenge.net/data/MOT17/) results obtained using [motmetrics4norfair](https://github.com/akator-de/norfair-enough/tree/main/demos/motmetrics4norfair) demo script on the `train` split. We used detections obtained with [ByteTrack's](https://github.com/ifzhang/ByteTrack) YOLOX object detection model.

| MOT17 Train |   IDF1 IDP IDR    | Rcll  | Prcn  |  MOTA MOTP  |
| :---------: | :---------------: | :---: | :---: | :---------: |
|  MOT17-02   | 61.3% 63.6% 59.0% | 86.8% | 93.5% | 79.9% 14.8% |
|  MOT17-04   | 93.3% 93.6% 93.0% | 98.6% | 99.3% | 97.9% 07.9% |
|  MOT17-05   | 77.8% 77.7% 77.8% | 85.9% | 85.8% | 71.2% 14.7% |
|  MOT17-09   | 65.0% 67.4% 62.9% | 90.3% | 96.8% | 86.8% 12.2% |
|  MOT17-10   | 70.2% 72.5% 68.1% | 87.3% | 93.0% | 80.1% 18.7% |
|  MOT17-11   | 80.2% 80.5% 80.0% | 93.0% | 93.6% | 86.4% 11.3% |
|  MOT17-13   | 79.0% 79.6% 78.4% | 90.6% | 92.0% | 82.4% 16.6% |
|   OVERALL   | 80.6% 81.8% 79.6% | 92.9% | 95.5% | 88.1% 11.9% |

| MOT20 Train |   IDF1 IDP IDR    | Rcll  | Prcn  |  MOTA MOTP  |
| :---------: | :---------------: | :---: | :---: | :---------: |
|  MOT20-01   | 85.9% 88.1% 83.8% | 93.4% | 98.2% | 91.5% 12.6% |
|  MOT20-02   | 72.8% 74.6% 71.0% | 93.2% | 97.9% | 91.0% 12.7% |
|  MOT20-03   | 93.0% 94.1% 92.0% | 96.1% | 98.3% | 94.4% 13.7% |
|  MOT20-05   | 87.9% 88.9% 87.0% | 96.0% | 98.1% | 94.1% 13.0% |
|   OVERALL   | 87.3% 88.4% 86.2% | 95.6% | 98.1% | 93.7% 13.2% |

## Citing

For citations in academic publications, please reference the original Norfair project. Export your desired citation format (BibTeX or other) from the [Zenodo entry](https://doi.org/10.5281/zenodo.5146253).

## License

Copyright © 2022, [Tryolabs](https://tryolabs.com) and © 2026, Akator GmbH and contributors. Released under the [BSD 3-Clause](https://github.com/akator-de/norfair-enough/blob/main/LICENSE).

This project is a fork of [tryolabs/norfair](https://github.com/tryolabs/norfair).
