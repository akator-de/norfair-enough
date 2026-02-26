# Code Review: norfair-enough

**Date:** 2026-02-26
**Scope:** Full codebase review covering core pipeline, supporting modules, drawing subpackage, tests, and CI/CD.

---

## Summary

The project is well-structured with clean abstractions (ABC + Factory pattern), good separation of concerns, and a thoughtful plugin system for distance functions and filters. However, there are several bugs, design issues, and significant test coverage gaps that should be addressed.

---

## 1. Critical Bugs

### CR-01: Missing f-string prefix in `video.py:273` — FIXED (PR #11)

```python
file_name = "camera_{self.camera}"  # BUG: not an f-string
```

Produces literal `camera_{self.camera}` instead of e.g. `camera_0`.

### CR-02: `draw_tracked_objects` returns `None` (`draw_points.py:186`) — FIXED (PR #11)

The function calls `_draw_points_alias(...)` but does not `return` the result. Callers writing `frame = draw_tracked_objects(...)` receive `None`.

### CR-03: NaN text position when all points are dead (`draw_points.py:163-164`) — FIXED (PR #11)

When all points are dead and `hide_dead_points=False`, `d.points[d.live_points].mean()` operates on an empty array, producing `NaN`. Text is drawn at an undefined position.

### CR-04: `lru_cache` on mutable NumPy array (`absolute_grid.py:11`) — FIXED (PR #11)

`_get_grid` returns a cached NumPy array. If `coord_transformations.abs_to_rel(points)` modifies the array in-place, the cache is corrupted for all subsequent calls.

### CR-05: Misplaced docstring in `metrics.py:78-82` — FIXED (PR #11)

```python
def update(self, predictions, frame_number=None):
    if frame_number is None:
        frame_number = self.frame_number
    """Write tracked object information..."""  # Never recognized as docstring
```

---

## 2. Medium Bugs

### CR-06: `distance_matrix.any()` skips zero distances (`tracker.py:305`) — FIXED (PR #11)

`distance_matrix.any()` returns `False` when all distances are exactly `0.0`, skipping the `current_min_distance` assignment. Should be `distance_matrix.size > 0`.

### CR-07: `hex_to_bgr` rejects uppercase hex (`color.py:28`) — FIXED (PR #12)

Regex `#[a-f0-9]{6}$` only matches lowercase. `"#FF0000"` raises `ValueError`.

### CR-08: Float downsample_ratio produces float dimensions (`video.py:249`) — FIXED (PR #12)

`//` with float `downsample_ratio` produces float, but `cv2.resize` expects integer dimensions.

### CR-09: Extension extraction fails for non-3-char extensions (`video.py:283`) — FIXED (PR #12)

`filename[-3:]` fails for `.webm`, `.mkv`, `.ts`, etc.

### CR-10: Non-portable path splitting (`video.py:271`) — FIXED (PR #12)

Hardcoded `/` instead of `os.path.basename()` breaks on Windows.

### CR-11: Division-by-zero fix can flip sign (`camera_motion.py:155-156`) — FIXED (PR #12)

Replacing zeros with `0.0000001` can change the sign of negative-approaching values.

### CR-12: File handle leak in `PredictionsTextFile` (`metrics.py:72`) — FIXED (PR #12)

File opened without context manager, never closed on early termination or exceptions.

### CR-13: Quadratic array growth (`metrics.py:228-235`) — FIXED (PR #12)

`np.vstack` in a loop creates a new array each iteration. Should accumulate in a list.

### CR-14: Typo in error message (`drawer.py:368`) — FIXED (PR #12)

`"Extecting"` should be `"Expecting"`.

### CR-15: Memory leak in `AbsolutePaths` (`path.py:258`) — FIXED (PR #12)

`self.past_points` for destroyed tracked objects is never cleaned up.

### CR-16: `draw_tracked_boxes` docstring references wrong function name — FIXED (PR #12)

References `draw_box` when the function is `draw_boxes`.

---

## 3. Design Issues

### CR-17: Mutable default argument (`tracker.py:90`) — FIXED (PR #13)

```python
filter_factory: FilterFactory = OptimizedKalmanFilterFactory()
```

Instance created once at class definition time and shared across all `Tracker` instances. Should use `None` with fallback in `__init__`.

### CR-18: `global_count` not thread-safe (`tracker.py:396`) — FIXED (PR #18)

`_TrackedObjectFactory.global_count` is a class variable incremented without synchronization. `Tracker.update()` is also not thread-safe. Should at minimum be documented.

### CR-19: Inconsistent warning mechanisms — FIXED (PR #13)

Three different methods used:
- `from logging import warning` (deprecated module-level function, in 4 files)
- `print()` in `distances.py:124` (in hot loop!)
- `warn_once()` from `utils.py`

Should unify to `logging.getLogger(__name__)`.

### CR-20: `assert` for input validation (`tracker.py:658,807`, `distances.py:379`) — FIXED (PR #13)

Assertions disabled with `python -O`. Should use `if ... raise ValueError(...)`.

### CR-21: Missing public API exports in `__init__.py` — FIXED (PR #19)

Not exported despite being public API:
- `TrackedObject`, `Distance`, `FilterFactory`, `ColorLike`
- All `camera_motion` classes (`MotionEstimator`, `HomographyTransformation`, etc.)
- All `metrics` classes

### CR-22: `__getattribute__` instead of `__getattr__` (`utils.py:71`) — FIXED (PR #13)

`DummyOpenCVImport.__getattribute__` blocks ALL attribute access including `__class__`, `__repr__` etc. `__getattr__` would be sufficient and preserve basic introspection.

### CR-23: No context manager for `Video` (`video.py:182-188`) — FIXED (PR #20)

Resources (VideoCapture, OutputVideo) only released on full iterator exhaustion. Early `break` causes resource leak.

### CR-24: `Detection` objects mutated by tracker (`tracker.py:530, 726`) — FIXED (PR #21)

Tracker sets `detection.age` on input `Detection` objects. Surprising side effect for callers.

### CR-25: Error message contradicts code (`tracker.py:121-126`) — FIXED (PR #13)

Message says `"should be larger than 0"` but code accepts `0`.

### CR-26: Fragile `hasattr` type dispatch (`distances.py:210-218`) — OPEN

Uses `hasattr(c, "points")` to distinguish `Detection` from `TrackedObject`. Should use `isinstance`.

### CR-27: `iou` only validates candidates, not objects (`distances.py:409-410`) — FIXED (PR #13)

`_validate_bboxes(candidates)` called but objects not validated.

---

## 4. Performance — OPEN

### CR-28: Greedy matching is O(n*m*min(n,m)) (`tracker.py:359-392`)

Each iteration does full `argmin()` + `min()` over the entire matrix. Acknowledged in code comment but unfixed.

### CR-29: `list.remove()` in loop (`tracker.py:344`)

O(n) per call, called O(k) times = O(n*k) total.

### CR-30: `ScalarDistance` uses Python double loop (`distances.py:120-127`)

Inherently slow for large numbers of candidates/objects. Warning exists but no vectorized alternative.

### CR-31: Redundant label iteration (`distances.py:188-226`)

Label masks computed but not used; inner loop re-compares labels manually.

### CR-32: `AbsolutePaths.draw` copies frame O(N*M) times (`path.py:233`)

`frame.copy()` inside inner loop over past points.

### CR-33: Iterative alpha blending is mathematically incorrect (`path.py:256`)

Compound blending doesn't produce the linear fade that `np.linspace(0.99, 0.01, max_history)` suggests.

---

## 5. Typos & Docstring Issues — ALL FIXED (PR #14)

| ID | File:Line | Issue | Status |
|---|---|---|---|
| CR-34 | `camera_motion.py:1` | `"stimation"` -> `"estimation"` | FIXED |
| CR-35 | `camera_motion.py:356-357` | Docstring example missing `import` keyword | FIXED |
| CR-36 | `path.py:107` | `"your"` -> `"you're"` | FIXED |
| CR-37 | `color.py:315` | `"pallete"` -> `"palette"` | FIXED |
| CR-38 | `drawer.py:284` | `"wheigthted"` -> `"weighted"` | FIXED |
| CR-39 | `draw_boxes.py:29` | Stray non-ASCII `'` at end of comment | FIXED |
| CR-40 | `filter.py:146-149` | Placeholder docstring `_description_` never filled | FIXED |
| CR-41 | `video.py:205-206` | Placeholder docstring `_description_` | FIXED |

---

## 6. Test Coverage Gaps — PARTIALLY FIXED (PR #15)

### Zero test coverage

| Module | Notes | Status |
|---|---|---|
| `camera_motion.py` | Complex logic, no tests at all | OPEN |
| `video.py` | Requires OpenCV | OPEN |
| `filter.py` (`NoFilterFactory`) | Publicly exported, never tested | FIXED |
| `drawing/draw_points.py` | No tests for actual drawing | OPEN |
| `drawing/draw_boxes.py` | No tests for actual drawing | OPEN |
| `drawing/path.py` | No tests | OPEN |
| `drawing/fixed_camera.py` | No tests | OPEN |
| `drawing/absolute_grid.py` | No tests | OPEN |
| `utils.py` | No direct tests | OPEN |

### Important untested scenarios

| Scenario | Location | Status |
|---|---|---|
| Label-based tracking | `ScalarDistance`/`VectorizedDistance` label matching | FIXED |
| `TrackedObject.estimate_velocity` | `tracker.py:568-576` | FIXED |
| `Detection.data`/`embedding` | `tracker.py` | OPEN |
| NaN in distance matrix | `tracker.py:299-302` | FIXED |
| `period` parameter | `tracker.py:update()` | FIXED |
| Multi-detection/multi-object distance matrices | Only 1x1 tested | FIXED |
| `NoFilterFactory` | `filter.py` | FIXED |

### Test quality issues

| Issue | Details | Status |
|---|---|---|
| `_TrackedObjectFactory.global_count` state leak | Class variable never reset between tests | FIXED |
| `test_drawing.py:241-243` | Enshrines typo `"Extecting"` instead of catching it | FIXED |
| `mot_metrics.py:14` | `sys.exit(0)` on NumPy 2.x masks incompatibility as success | FIXED |
| Nested `for` loops vs parametrization | `test_tracker.py:47`, `test_tracker.py:220` | OPEN |

---

## 7. CI/CD Issues — ALL FIXED (PR #16, #17)

| ID | Issue | File | Status |
|---|---|---|---|
| CR-42 | Lint only runs on PRs, not push to main | `lint.yml:3` | FIXED |
| CR-43 | Release job doesn't depend on build/install verification | `ci.yml:155` | FIXED |
| CR-44 | Coverage job doesn't install optional dependencies | `ci.yml:82` | FIXED |
| CR-45 | No macOS/Windows CI | `ci.yml` | OPEN |
| CR-46 | Inconsistent Ubuntu versions (22.04 vs 24.04) | `dependabot-auto-merge.yml`, `labeler.yml` | FIXED |
| CR-47 | `upload-artifact@v6` / `download-artifact@v7` mismatch | `ci.yml:118,137` | FIXED |

---

## Remaining Open Items

| CR | Category | Summary |
|---|---|---|
| CR-26 | Design | Fragile `hasattr` type dispatch in distances |
| CR-28–33 | Performance | Greedy matching, list.remove, ScalarDistance loop, label iteration, frame copies, alpha blending |
| CR-45 | CI | No macOS/Windows CI |
