# Filter

Each [`TrackedObject`][norfair.tracker.TrackedObject] uses a Kalman filter to
smooth its estimated position and velocity over time. The `filter_factory`
argument on [`Tracker`][norfair.tracker.Tracker] decides which filter is built
for every new track.

Norfair ships with two filter factories:

- [`OptimizedKalmanFilterFactory`][norfair.filter.OptimizedKalmanFilterFactory] —
  the default. A NumPy-only constant-velocity Kalman filter tuned for speed.
  Use this unless you need custom dynamics.
- [`FilterPyKalmanFilterFactory`][norfair.filter.FilterPyKalmanFilterFactory] —
  backed by the [`filterpy`](https://github.com/rlabbe/filterpy) library.
  Slower but makes it easy to swap in a custom `filterpy` Kalman filter if you
  need a different motion model.

Both expose the same knobs (`R` — measurement noise, `Q` — process noise) so
you can trade smoothness against responsiveness without changing tracker code:

- Higher `R` or lower `Q` → the filter trusts the estimate more → smoother,
  but slower to react to real motion.
- Lower `R` or higher `Q` → the filter trusts the measurement more → twitchier
  but more responsive.

## Example

```python
from norfair import Tracker
from norfair.filter import OptimizedKalmanFilterFactory

tracker = Tracker(
    distance_function="euclidean",
    distance_threshold=50,
    # Increase R to reduce jitter on noisy detectors.
    filter_factory=OptimizedKalmanFilterFactory(R=8.0, Q=0.1),
)
```

To disable filtering entirely (e.g. for debugging raw detections end-to-end),
pass a [`NoFilterFactory`][norfair.filter.NoFilterFactory] instead.

## API

::: norfair.filter
    options:
        show_root_heading: false
        members:
            - FilterFactory
            - FilterPyKalmanFilterFactory
            - NoFilterFactory
            - OptimizedKalmanFilterFactory

## See also

- [Tracker](tracker.md) — accepts the `filter_factory` used here.
- [Getting Started](../getting_started.md#tracking-issues) — practical tips on
  tuning `R` and `Q` when the tracker jitters or lags.
