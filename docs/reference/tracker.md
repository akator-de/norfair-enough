# Tracker

The tracker module is the core of Norfair. It takes per-frame detections from
any object or keypoint detector and maintains a set of `TrackedObject`s with
stable ids across frames, using a configurable distance function and a Kalman
filter to smooth the estimated state.

Most pipelines only need to interact with three things from this module:

- [`Detection`][norfair.tracker.Detection] — wraps the points/bbox produced by your detector for a single frame.
- [`Tracker`][norfair.tracker.Tracker] — matches detections to existing tracks, spawns new ones, and ages out stale ones.
- [`TrackedObject`][norfair.tracker.TrackedObject] — what `Tracker.update()` returns; the stable, id-carrying representation of an object across frames.

## Minimal example

```python
from norfair import Detection, Tracker, Video, draw_points

tracker = Tracker(
    distance_function="euclidean",
    distance_threshold=50,
    initialization_delay=3,
    hit_counter_max=15,
)

with Video(input_path="video.mp4") as video:
    for frame in video:
        # my_detector returns a list of (N, 2) arrays — one per object
        detections = [Detection(points=p) for p in my_detector(frame)]
        tracked_objects = tracker.update(detections=detections)
        draw_points(frame, tracked_objects)
        video.write(frame)
```

A few things worth knowing:

- `Tracker.update()` must be called **once per frame**, in order. Norfair is an
  online tracker — it never looks at future frames.
- Pass `period=N` if you only run the detector every `N` frames but still call
  `update()` every frame so the filter can keep predicting.
- For moving cameras, pass the `coord_transformations` returned by
  [`MotionEstimator`][norfair.camera_motion.MotionEstimator] into `update()` —
  see [Camera Motion](camera_motion.md).

## Choosing a distance function

The `distance_function` argument decides how detections are matched to tracks
each frame. Norfair ships with several built-ins that you can select by name
(`"euclidean"`, `"mean_euclidean"`, `"mean_manhattan"`, `"frobenius"`,
`"iou"`, `"iou_opt"`) plus any metric supported by `scipy.spatial.distance.cdist`.
You can also pass your own `Callable` or a subclass of
[`Distance`][norfair.distances.Distance] for appearance-aware matching.

See [Distances](distances.md) for the full list and how to build custom ones.

## API

::: norfair.tracker
    options:
        show_root_heading: false
        members:
            - Detection
            - Tracker
            - TrackedObject

## See also

- [Distances](distances.md) — how detections are matched to tracks.
- [Filter](filter.md) — Kalman filter factories used to smooth state estimates.
- [Drawing](drawing.md) — rendering the returned `TrackedObject`s on frames.
- [Camera Motion](camera_motion.md) — handling moving cameras.
