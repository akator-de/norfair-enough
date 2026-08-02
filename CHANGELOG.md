# Changelog

All notable changes to norfair-enough will be documented in this file.

This fork is based on [tryolabs/norfair](https://github.com/tryolabs/norfair) v2.3.0. Published on PyPI as [`norfair-enough`](https://pypi.org/project/norfair-enough/).

## [Unreleased]

## [2.4.3] - 2026-08-02

### Added
- Synthetic end-to-end scenario tests for the tracking pipeline (#124)
- Docstrings for the full public API (#84)
- `numpy.typing.NDArray` annotations for differentiated array types (#85)

### Fixed
- `camera_motion`: raise `ValueError` on a singular homography instead of propagating an unusable matrix (#105)
- `camera_motion`: prevent view aliasing in `TranslationTransformationGetter` (#106)
- `filter`: clamp `vel_variance` to non-negative after update (#107)
- `fixed_camera`: clip before casting the attenuated background to `uint8`, avoiding overflow (#108)
- `distances`: group `VectorizedDistance` labels by identity instead of `str()`, so a `None` label no longer collides with the literal string `"None"` (#109)
- Correct the indentation of the tracking-loop example in the Getting Started guide

### Performance
- `drawing`: reuse the overlay scratch buffer in `AbsolutePaths.draw` (#122)
- `drawing`: `deque(maxlen=...)` for `AbsolutePaths` history instead of `list.insert(0, ...)` (#110)
- `kalman_filter`: compute the Kalman gain via `np.linalg.solve` instead of `np.linalg.inv` (#121)
- `filter`: `np.full` for `OptimizedKalmanFilter` variance buffers (#112)
- `filter`: operator form instead of `np.multiply`/`np.divide` in `update` (#113)
- `metrics`: build `Accumulators.paths` as a list instead of an O(n²) `np.hstack` loop (#111)
- `tracker`: identity set for `TrackedObject` removal (#116)
- `video`: throttle progress-bar refresh to ~20 Hz (#119)

### Changed
- **`opencv-python` is now required as `>=4.10.0.84,<6.0.0`, adding support for OpenCV 5.0.** The floor moved up because opencv-python releases before 4.10.0.84 are built against the NumPy 1.x C ABI and fail to import under NumPy 2.x, which this package permits (#118)
- Dependencies and tooling refreshed: ruff 0.16.1, pyrefly 1.2.0, pytest 9.1.1, and all locked versions updated
- GitHub Actions updated: checkout v7, setup-python v7, setup-uv v9, codecov-action v7, labeler v7
- `drawer`: explicit int conversion for rectangle corners (#115)
- `filter`: drop redundant `expand_dims().T` in favour of `reshape(-1, 1)` (#114)
- `utils`: drop the `rich.print` import that shadowed the built-in (#117)
- Remove the MOT17 integration test and make the MOT metrics CI job non-blocking (#123)

## [2.4.2] - 2026-04-16

### Fixed
- Fix `hit_counter` not set to `hit_counter_max` on initialization, causing newly tracked objects to exit after a few frames without detection (#82). Upstream: [tryolabs/norfair#337](https://github.com/tryolabs/norfair/issues/337)

## [2.4.1] - 2026-04-14

### Added
- Comprehensive test coverage for `camera_motion`, drawing subpackage, `Video`, and utils (#76)
- Add CodeRabbit-generated unit tests for `Video`, tracker, and package exports (#26)
- Add CodeRabbit configuration for automated PR reviews (#3)

### Changed
- Modernize demos: replace 12 outdated demos with Ultralytics YOLO/Pose/Camera-motion, add Dockerfiles with non-root user (#73)
- Consistent naming: use "Norfair Enough" instead of bare "Norfair" throughout docs and demos (#75)
- Modernize README with updated badges, comparison section, and motivation (#74)
- Optimize greedy matching to avoid O(n²) `list.remove()` (#64)
- Add OS matrix (Linux/macOS/Windows), fix `pull_request_target` security, packaging polish (#65)
- Replace `print()` with `logging` in metrics module
- Add maintainer metadata and project URLs to `pyproject.toml`
- Document `Detection` mutation behavior in `Tracker.update()` (`.absolute_points`, `.age`) and `Detection` docstrings (#22)
- Replace fragile `hasattr` type dispatch with `isinstance` in vectorized distance functions (#23)
- Add context manager support and `close()` method to `Video` for reliable resource cleanup (#20)
- Protect `TrackedObject.global_id` counter with a lock for thread safety (#18)
- Export all public API symbols (`TrackedObject`, `Distance`, `FilterFactory`, `ColorLike`, camera motion and metrics classes) from the top-level package
- Replace mutable default argument in `Tracker.__init__`
- Unify logging to use `logging.getLogger(__name__)` across all modules
- Replace `assert` with `ValueError` for input validation
- Use `__getattr__` instead of `__getattribute__` in dummy import classes
- Validate both candidates and objects in `iou` distance function
- Improve test coverage for `NoFilterFactory`, label matching, and drawing (#15)
- Improve CI pipeline: lint on push, release gates, coverage with extras (#16)
- Update demos to use the `norfair-enough` package (#4)
- Bump GitHub Actions dependencies: `actions/download-artifact` v6→v7, `actions/labeler` v5→v6 (#17)

### Fixed
- Fix MOT20 benchmark link pointing to MOT17 in README
- Guard Kalman filter against singular/ill-conditioned innovation covariance (#66)
- Guard distance functions against NaN values and zero division (#67)
- Guard `cv2.findHomography()` against returning `None` (#68)
- Guard drawing against NaN/Inf coordinates and enforce rectangle corner ordering (#69)
- Prevent `Detection.age` mutation and deep-copy mutable refs in `merge()` (#70)
- Add input validation to `get_cutout` and `print_objects_as_table` (#71)
- Correct broken anchor and MediaPipe link in README (#72)
- Add context manager support and close file handles in metrics (#63)
- Fix missing f-string for camera output filename (`video.py`)
- Fix `draw_tracked_objects` not returning the frame
- Fix NaN text position when all tracked points are dead
- Fix `lru_cache` on mutable NumPy array in grid drawing
- Fix misplaced docstring in `PredictionsTextFile.update`
- Fix `distance_matrix.any()` incorrectly skipping zero distances
- Fix `hex_to_bgr` rejecting uppercase hex color strings
- Fix float dimensions from `downsample_ratio` in video output
- Fix non-portable path handling in video output filename
- Fix file extension extraction in video codec selection
- Fix division-by-zero workaround in homography transformation
- Fix file handle leak in `PredictionsTextFile`
- Fix quadratic array growth in `Accumulators.update`
- Fix typo "Extecting" in `Drawable` error message
- Fix memory leak for destroyed objects in `AbsolutePaths`
- Fix wrong function reference in `draw_tracked_boxes` docstring
- Fix various typos and placeholder docstrings (#14)

## [2.4.0] - 2025-02-26

### Breaking Changes

- **Drop Python < 3.10**: Minimum required Python version is now 3.10 (upstream PR [#335](https://github.com/tryolabs/norfair/pull/335))
- **Remove filterpy dependency**: The deprecated `filterpy` library has been replaced by an internal Kalman filter implementation (upstream PR [#330](https://github.com/tryolabs/norfair/pull/330)). `FilterPyKalmanFilterFactory` now uses the internal implementation and no longer requires filterpy to be installed.

### Added

- **NumPy 2.x support**: Norfair now works with both NumPy 1.23+ and NumPy 2.x. The MOT metrics test environment pins NumPy < 2 due to motmetrics incompatibility (upstream PR [#335](https://github.com/tryolabs/norfair/pull/335))
- **Internal Kalman filter** (`norfair/kalman_filter.py`): Self-contained implementation based on the original FilterPy `KalmanFilter` class (MIT, Roger R. Labbe Jr.), adapted to Norfair's architecture (upstream PR [#330](https://github.com/tryolabs/norfair/pull/330))
- **`TrackedObject.scores` attribute**: Tracked objects now expose the scores from their last matched detection. Previously this was always `None` (upstream PR [#311](https://github.com/tryolabs/norfair/pull/311))
- **Single score for Detection**: `Detection(scores=...)` now accepts a single `float` or `int` in addition to `np.ndarray`. A scalar value is automatically broadcast to all points (upstream PR [#295](https://github.com/tryolabs/norfair/pull/295))

### Fixed

- **ReID track pruning bug**: `reid_hit_counter` is now reset to `None` when a tracked object is successfully matched to a detection. Previously, if `hit_counter` had dropped to 0 and `reid_hit_counter` was activated, matching the object again would not clear the reid countdown, causing the object to be incorrectly pruned after `reid_hit_counter_max` frames despite being actively tracked (upstream issue [#325](https://github.com/tryolabs/norfair/issues/325), PR [#326](https://github.com/tryolabs/norfair/pull/326))

### Changed

- **Modernized dependencies**: `scipy >= 1.13.1`, `rich ^14.0.1`, `numpy >= 1.23.0`
- **Tox configuration**: Uses `poetry-plugin-export` for dependency installation; MOT metrics isolated in dedicated `mot-py313` environment with NumPy < 2
