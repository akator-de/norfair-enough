# Video

Optional helper for reading video files or a live camera feed, writing the
annotated output, and showing a progress bar during processing. It is a thin
wrapper around OpenCV's `VideoCapture` / `VideoWriter` — use it for quick
pipelines, and reach for OpenCV directly when you need more control.

To use this module install Norfair with the `video` extra:

```bash
pip install 'norfair-enough[video]'
```

## Context manager usage (recommended)

[`Video`][norfair.video.Video] implements the context-manager protocol so file
handles and writers are released deterministically — even if the loop is
interrupted by an exception or an early `break`:

```python
from norfair import Detection, Tracker, Video, draw_points

tracker = Tracker(distance_function="euclidean", distance_threshold=50)

with Video(input_path="video.mp4", output_path="out.mp4") as video:
    for frame in video:
        detections = [Detection(points=p) for p in my_detector(frame)]
        tracked_objects = tracker.update(detections=detections)
        draw_points(frame, tracked_objects)
        video.write(frame)
```

## Reading from a webcam

Pass `camera=<device id>` instead of `input_path`:

```python
with Video(camera=0) as video:
    for frame in video:
        ...
```

`camera` and `input_path` are mutually exclusive — you must pick one.

## API

::: norfair.video
    options:
        show_root_heading: false

## See also

- [Drawing](drawing.md) — helpers to annotate frames before writing them.
- [Tracker](tracker.md) — consumes the frames produced here.
