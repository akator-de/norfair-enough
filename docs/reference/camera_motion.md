# Camera Motion

When the camera itself moves, the apparent motion of tracked objects becomes
a mix of "the object moved" and "the camera moved". Norfair's camera motion
module estimates the camera's frame-to-frame transformation from optical flow
and lets the [`Tracker`][norfair.tracker.Tracker] reason about positions in a
stable world coordinate system.

The main entry point is [`MotionEstimator`][norfair.camera_motion.MotionEstimator].
It returns a [`CoordinatesTransformation`][norfair.camera_motion.CoordinatesTransformation]
for each frame which you pass to `Tracker.update(coord_transformations=...)`.
Two concrete transformation families are provided:

- [`TranslationTransformation`][norfair.camera_motion.TranslationTransformation] /
  [`TranslationTransformationGetter`][norfair.camera_motion.TranslationTransformationGetter] —
  cheap, good enough for pure pan/tilt.
- [`HomographyTransformation`][norfair.camera_motion.HomographyTransformation] /
  [`HomographyTransformationGetter`][norfair.camera_motion.HomographyTransformationGetter] —
  full 8-DoF homography, handles rotation, zoom, and perspective. This is the
  default.

## Example

```python
from norfair import Detection, Tracker, Video, draw_points
from norfair.camera_motion import MotionEstimator

tracker = Tracker(distance_function="euclidean", distance_threshold=50)
motion_estimator = MotionEstimator()

with Video(input_path="video.mp4") as video:
    for frame in video:
        coord_transformation = motion_estimator.update(frame)

        detections = [Detection(points=p) for p in my_detector(frame)]
        tracked_objects = tracker.update(
            detections=detections,
            coord_transformations=coord_transformation,
        )

        draw_points(frame, tracked_objects)
        video.write(frame)
```

You can pass a `mask` to `motion_estimator.update()` to ignore image regions
that contain the tracked objects themselves (which would otherwise pollute the
optical-flow estimate). A common pattern is to build the mask from the previous
frame's detections or tracked objects.

For debugging, set `MotionEstimator(draw_flow=True)` to overlay the sampled
optical-flow vectors on the frame, and use
[`draw_absolute_grid`][norfair.drawing.draw_absolute_grid] together with
[`FixedCamera`][norfair.drawing.FixedCamera] to visualize the recovered
world coordinate system.

## API

::: norfair.camera_motion
    options:
        show_root_heading: false

## See also

- [Tracker](tracker.md) — consumes the `coord_transformations` returned here.
- [Drawing](drawing.md) — [`FixedCamera`][norfair.drawing.FixedCamera] and
  [`draw_absolute_grid`][norfair.drawing.draw_absolute_grid] pair with
  `MotionEstimator` for visualization.
