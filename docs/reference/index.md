# API Reference

The Norfair API is split into a handful of focused modules. Most tracking
pipelines only touch a few of them: you wrap your detector's output in
[`Detection`][norfair.tracker.Detection]s, feed them to a
[`Tracker`][norfair.tracker.Tracker], and optionally draw the results on top of
the original frames.

## Modules

| Module | Purpose |
|---|---|
| [Tracker](tracker.md) | `Tracker`, `Detection`, `TrackedObject` — the core online tracker. |
| [Distances](distances.md) | Built-in and custom distance functions used to match detections to tracks. |
| [Filter](filter.md) | Kalman filter factories that smooth the estimated state of each track. |
| [Drawing](drawing.md) | Helpers to draw detections, tracked objects, paths, and grids on frames. |
| [Video](video.md) | Optional helper for reading video files / camera feeds and writing output. |
| [Camera Motion](camera_motion.md) | `MotionEstimator` and coordinate transformations for moving-camera scenarios. |
| [Metrics](metrics.md) | MOT Challenge accumulators and file parsers for evaluation. |
| [Utils](utils.md) | Small utilities (`get_cutout`, `print_objects_as_table`). |

## Tracking in 5 lines

The smallest useful pipeline only needs [`Tracker`][norfair.tracker.Tracker],
[`Detection`][norfair.tracker.Detection], and — optionally —
[`draw_points`][norfair.drawing.draw_points] to visualize the result:

```python
from norfair import Detection, Tracker, Video, draw_points

tracker = Tracker(distance_function="euclidean", distance_threshold=50)
with Video(input_path="video.mp4") as video:
    for frame in video:
        detections = [Detection(points) for points in my_detector(frame)]
        tracked_objects = tracker.update(detections=detections)
        draw_points(frame, tracked_objects)
        video.write(frame)
```

For an end-to-end walkthrough (installation, detector integration, debugging
tips) see [Getting Started](../getting_started.md).
