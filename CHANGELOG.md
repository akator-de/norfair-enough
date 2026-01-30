# Changelog

All notable changes to norfair-enough will be documented in this file.

This fork is based on [tryolabs/norfair](https://github.com/tryolabs/norfair) v2.3.0. Published on PyPI as [`norfair-enough`](https://pypi.org/project/norfair-enough/).

## [Unreleased]

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
