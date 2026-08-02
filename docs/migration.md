# Migration Guide: tryolabs/norfair → norfair-enough

`norfair-enough` is a fork of [`tryolabs/norfair`](https://github.com/tryolabs/norfair),
based on upstream v2.3.0. This guide describes what you need to change when moving an existing
project from the upstream package to `norfair-enough`.

!!! warning "This fork is no longer maintained"

    v2.4.3 (August 2026) is the final release and the repository is archived. The package
    remains installable and working, but nothing further will be released. This guide is kept
    for anyone migrating onto that final version, or forking it to carry it forward.

## TL;DR

- Uninstall `norfair`, install `norfair-enough`.
- The import path stays the same: `from norfair import ...`.
- Make sure you are on Python 3.10 or newer.
- Most code continues to work unchanged; a few APIs have been deprecated or behavior has been
  tightened.

## Installation

Replace the upstream package with the fork:

```bash
pip uninstall norfair
pip install norfair-enough
```

!!! warning "Do not install both packages at the same time"
    Both `norfair` and `norfair-enough` ship the same top-level Python module (`norfair`).
    Installing both into the same environment will lead to undefined import behavior.
    Always uninstall one before installing the other.

Imports do **not** change — `norfair-enough` keeps the original module name:

```python
from norfair import Detection, Tracker, draw_points
```

If you install from source, use the fork's repository URL:

```bash
pip install "git+https://github.com/akator-de/norfair-enough.git@main#egg=norfair-enough"
```

## Python version

`norfair-enough` requires **Python 3.10 or newer**.

Upstream `norfair` historically supported earlier Python versions (3.8/3.9).
If your project is still on Python 3.8 or 3.9 you will need to upgrade your interpreter
before migrating. The fork follows the upstream decision (v2.4.0) to drop older Python
versions — see [CHANGELOG.md](https://github.com/akator-de/norfair-enough/blob/main/CHANGELOG.md).

The CI matrix for `norfair-enough` covers Python 3.10 through 3.13.

## Dependency changes

- **`filterpy` is no longer a dependency.** The fork ships an internal Kalman filter
  implementation derived from the original FilterPy `KalmanFilter` class (MIT, Roger R. Labbe Jr.).
  `FilterPyKalmanFilterFactory` still exists for backwards compatibility but now uses the
  internal implementation — you can remove `filterpy` from your own requirements.
- **NumPy 2.x is supported** alongside NumPy 1.23+. No code change required on your side.

## Deprecated drawing APIs

The following drawing helpers are **deprecated** in `norfair-enough` but still work. They now
emit a deprecation warning on first use. Migrate at your convenience:

| Deprecated                 | Replacement     |
| -------------------------- | --------------- |
| `draw_tracked_objects(...)` | `draw_points(...)` |
| `draw_tracked_boxes(...)`   | `draw_boxes(...)`  |

Both replacements accept tracked objects directly, so most call sites only need a rename.

Additional deprecated parameters on `draw_points`:

- `color_by_label=True` → use `color="by_label"`
- `detections=...`      → use `drawables=...`
- `label_size=...`      → use `text_size=...`

Additional deprecated parameters on `draw_boxes`:

- `color_by_label=True` → use `color="by_label"`
- `detections=...`      → use `drawables=...`
- `label_size=...`      → use `text_size=...`
- `random_color=True`   → use `color="random"`
- `line_color=...`      → use `color=...`
- `line_width=...`      → use `thickness=...`

See the [Drawing reference][norfair.drawing] for the full current API.

## Behavior changes

### `Tracker.update()` may mutate `Detection` objects -- behavior now documented

In both upstream `norfair` and `norfair-enough`, `Tracker.update()` may mutate attributes
on the `Detection` instances you pass in (for example `detection.absolute_points` when
`coord_transformations` is provided, and `detection.age` when detections are mapped to
tracked objects).

The difference in `norfair-enough` is that this mutation behavior is now documented
explicitly in the docstrings for `Tracker.update()` and `Detection`. If your code reuses
`Detection` objects after calling `Tracker.update()` -- for drawing, logging, or further
processing -- be aware of these mutations and review your code accordingly. In most
real-world pipelines this is invisible.

### Stricter input validation

- `Tracker` and related APIs now raise `ValueError` for invalid input that previously
  triggered `assert` statements. If you run Python with `-O`, the previous assertions
  were silently skipped; now the checks always run.
- The `iou` distance function validates both candidates and objects (previously only one
  side was checked).

### Other notable fixes

These are bug fixes relative to upstream v2.3.0. If your code accidentally depended on
the buggy behavior you may see different results:

- `draw_tracked_objects` now returns the frame (previously returned `None`).
- `distance_matrix.any()` no longer incorrectly skips zero distances during matching.
- `hex_to_bgr` now accepts uppercase hex color strings.
- `PredictionsTextFile` fixes file-handle leaks.
- `AbsolutePaths` fixes memory leaks for destroyed objects.
- `Accumulators.update` fixes quadratic array growth.
- `Video` output filenames now handle paths and file extensions portably; it also supports
  the context manager protocol (`with Video(...) as v:`) and exposes a `close()` method
  for reliable resource cleanup.
- `TrackedObject.global_id` counter is now thread-safe.

For the full list of fixes and changes, see
[CHANGELOG.md](https://github.com/akator-de/norfair-enough/blob/main/CHANGELOG.md).

## Newly exported symbols

The following symbols are now exported directly from the top-level `norfair` package and
no longer need to be imported from submodules:

- `TrackedObject`
- `Distance`
- `FilterFactory`
- `ColorLike`
- camera motion classes
- metrics classes

Old imports continue to work.

## Repository differences

- **Demos**: the fork ships modernized demos under `demos/`. Some have been updated to
  newer model versions and dependencies.
- **Tooling**: `norfair-enough` uses [`ruff`](https://docs.astral.sh/ruff/) for linting and
  `pytest` for tests, with a CI matrix covering Python 3.10–3.13.
- **Packaging**: published on PyPI as
  [`norfair-enough`](https://pypi.org/project/norfair-enough/).

## Staying on upstream `tryolabs/norfair`

Upstream `tryolabs/norfair` has had no commits since April 2025. If you prefer to stay on it,
you can continue to install it from
[its repository](https://github.com/tryolabs/norfair) or PyPI — but be aware that bug fixes
and new Python/NumPy compatibility work will not flow back.

Both projects are now dormant. `norfair-enough` v2.4.3 carries the fixes listed above and
requires Python 3.10+ (tested through 3.13), NumPy 2.x and OpenCV 5, so it is the
further-along of the two — but if you need something under active development, the
alternatives listed in the
[README](https://github.com/akator-de/norfair-enough#comparison-to-other-trackers) are the
better starting point.
