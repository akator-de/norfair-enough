# Distances

A **distance function** decides, for a given frame, how close each incoming
`Detection` is to every existing `TrackedObject`. The
[`Tracker`][norfair.tracker.Tracker] then uses these pairwise distances to
solve the matching problem (small distance → likely the same object).

Norfair ships with a handful of built-in distances, plus utilities for
building your own — either scalar (one detection / one track at a time) or
vectorized (all pairs at once, using NumPy).

## Built-in distances

You can pass any of these to `Tracker(distance_function=...)` by name:

| Name | Best for | Notes |
|---|---|---|
| `"euclidean"` | Single-point detections (centroids, keypoints). | Alias for `frobenius`. |
| `"mean_euclidean"` | Multi-point detections (keypoints, polygons). | Averages per-point Euclidean distance. |
| `"mean_manhattan"` | Multi-point detections, cheaper than Euclidean. | Averages per-point L1 distance. |
| `"frobenius"` | Flattened L2 between all points. | Same as `"euclidean"` for single-point detections. |
| `"iou"` | Bounding boxes. | `1 - IoU`, so smaller is better. Requires `(2, 2)` top-left / bottom-right point arrays. |
| `"iou_opt"` | Bounding boxes, large detection counts. | Vectorized, faster variant of `"iou"`. |

Any name accepted by
[`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
also works and is wrapped automatically as a
[`ScipyDistance`][norfair.distances.ScipyDistance].

```python
from norfair import Tracker

# Centroid tracking with plain Euclidean distance.
tracker = Tracker(distance_function="euclidean", distance_threshold=50)

# Bounding-box tracking with IoU. Threshold is in `1 - IoU` space,
# so 0.7 means "match if IoU >= 0.3".
bbox_tracker = Tracker(distance_function="iou", distance_threshold=0.7)
```

## Parameterized distances

Two factory helpers produce distances tailored to your data:

- [`create_normalized_mean_euclidean_distance`][norfair.distances.create_normalized_mean_euclidean_distance] —
  Mean Euclidean distance normalized by the frame size, so the same
  `distance_threshold` works across resolutions.
- [`create_keypoints_voting_distance`][norfair.distances.create_keypoints_voting_distance] —
  Counts how many keypoints are close enough to vote "same object"; useful for
  pose / keypoint trackers where detectors return many noisy points per object.

```python
from norfair import Tracker
from norfair.distances import create_normalized_mean_euclidean_distance

distance = create_normalized_mean_euclidean_distance(height=1080, width=1920)
tracker = Tracker(distance_function=distance, distance_threshold=0.05)
```

## Custom distances

For appearance-aware tracking (embeddings, ReID, color histograms, …) you can
pass any `Callable[[Detection, TrackedObject], float]` as
`distance_function`, or — for bulk performance — subclass
[`VectorizedDistance`][norfair.distances.VectorizedDistance] and compute all
pairs at once.

## API

::: norfair.distances
    options:
        show_root_heading: false

## See also

- [Tracker](tracker.md) — how the distance is consumed each frame.
- [Filter](filter.md) — the Kalman filter that produces the `estimate` used by
  most distance functions.
