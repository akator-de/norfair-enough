# Drawing

Lightweight drawing helpers built on top of OpenCV. They take a frame
(a `numpy.ndarray` in BGR order, as returned by [`Video`][norfair.video.Video])
and draw detections, tracked objects, paths, or reference grids **in place**.

The drawing helpers accept both `Detection`s and `TrackedObject`s, so you can
use the same call to visualize raw detector output and the final tracker
output on the same frame.

## Key functions

- [`draw_points`][norfair.drawing.draw_points] — draw points (keypoints,
  centroids) with optional labels, ids, and scores. This is the modern
  replacement for `draw_tracked_objects` and works with both detections and
  tracked objects.
- [`draw_boxes`][norfair.drawing.draw_boxes] — draw axis-aligned bounding
  boxes. Expects each drawable to carry its two corner points `(top_left,
  bottom_right)`. Replaces `draw_tracked_boxes`.
- [`Paths`][norfair.drawing.Paths] / [`AbsolutePaths`][norfair.drawing.AbsolutePaths] —
  trail-style visualization of where each tracked object has been.
- [`FixedCamera`][norfair.drawing.FixedCamera] — compensates for camera motion
  when visualizing, pairs with [`MotionEstimator`][norfair.camera_motion.MotionEstimator].
- [`draw_absolute_grid`][norfair.drawing.draw_absolute_grid] — draws a reference
  grid in world coordinates, useful for debugging camera motion estimation.
- [`Color`][norfair.drawing.Color] / [`Palette`][norfair.drawing.Palette] —
  named colors and palette configuration for `color="by_id"` / `"by_label"`.

## Example

```python
from norfair import Detection, Tracker, Video, draw_boxes, draw_points

tracker = Tracker(distance_function="euclidean", distance_threshold=50)

with Video(input_path="video.mp4") as video:
    for frame in video:
        raw = my_detector(frame)
        detections = [Detection(points=p) for p in raw]
        tracked_objects = tracker.update(detections=detections)

        # Draw raw detections in a fixed color so they stand out during debugging.
        draw_points(frame, detections, color="red", draw_ids=False)

        # Draw the tracker output with per-id colors and labels.
        draw_points(frame, tracked_objects, color="by_id")

        video.write(frame)
```

To draw bounding boxes instead, build each `Detection` with a `(2, 2)` array
of `[[x1, y1], [x2, y2]]` corners and call
[`draw_boxes`][norfair.drawing.draw_boxes] instead of `draw_points`.

## API

::: norfair.drawing.draw_points
    options:
        show_root_heading: true
::: norfair.drawing.draw_boxes
    options:
        show_root_heading: true
::: norfair.drawing.color
    options:
        show_root_heading: true
::: norfair.drawing.path
    options:
        show_root_heading: true
::: norfair.drawing.fixed_camera
    options:
        show_root_heading: true
::: norfair.drawing.absolute_grid
    options:
        show_root_heading: true

## See also

- [Tracker](tracker.md) — produces the `TrackedObject`s you draw.
- [Camera Motion](camera_motion.md) — pair [`FixedCamera`][norfair.drawing.FixedCamera]
  and [`draw_absolute_grid`][norfair.drawing.draw_absolute_grid] with a
  `MotionEstimator`.
