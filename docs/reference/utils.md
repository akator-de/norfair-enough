# Utils

A small grab-bag of helpers that don't fit anywhere else. Only two are part
of the public API:

- [`get_cutout`][norfair.utils.get_cutout] — crops the axis-aligned bounding
  rectangle around a set of points out of an image. Handy for feeding a patch
  into a ReID / appearance model to build your own distance function.
- [`print_objects_as_table`][norfair.utils.print_objects_as_table] — pretty-prints
  the current [`TrackedObject`][norfair.tracker.TrackedObject]s as a Rich table
  (id, age, hit counter, last distance, init id). Useful for quickly debugging
  why the tracker is — or isn't — matching things.

## Example

```python
from norfair import Detection, Tracker, Video, get_cutout, print_objects_as_table

tracker = Tracker(distance_function="euclidean", distance_threshold=50)

with Video(input_path="video.mp4") as video:
    for frame in video:
        detections = [Detection(points=p) for p in my_detector(frame)]
        tracked_objects = tracker.update(detections=detections)

        # Debug: dump the current track table.
        print_objects_as_table(tracked_objects)

        # Grab an appearance patch for the first tracked object.
        if tracked_objects:
            patch = get_cutout(tracked_objects[0].estimate, frame)
```

## API

::: norfair.utils
    options:
        show_root_heading: false
        members:
            - get_cutout
            - print_objects_as_table

## See also

- [Tracker](tracker.md) — source of the `TrackedObject`s inspected here.
- [Distances](distances.md) — `get_cutout` pairs well with a custom appearance
  distance.
