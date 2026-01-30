# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

norfair-enough is a maintained fork of [tryolabs/norfair](https://github.com/tryolabs/norfair), a lightweight Python library for real-time multi-object tracking. It adds tracking to any detector using Kalman filters and configurable distance functions. Published on PyPI as `norfair-enough`, but the Python import remains `norfair`.

## Development Setup

Uses **Poetry** for dependency management. The user prefers **uv** for running Python projects.

```bash
poetry install --all-extras    # install with all optional deps (video, metrics)
```

## Common Commands

### Testing
```bash
# Run unit tests (excludes MOT metrics integration test)
poetry run pytest -v tests/ --ignore=tests/mot_metrics.py

# Run a single test file
poetry run pytest -v tests/test_tracker.py

# Run a single test by name
poetry run pytest -v tests/test_tracker.py -k "test_name"

# Run with coverage
poetry run pytest -v tests/ --ignore=tests/mot_metrics.py --cov=norfair

# Run full test matrix via tox (Python 3.10-3.13)
poetry run tox
```

### Linting & Formatting
```bash
poetry run black --check --diff .       # check formatting
poetry run black .                       # auto-format
poetry run isort --check --diff .        # check import order
poetry run isort .                       # fix imports
poetry run pylint -E norfair/**/*.py     # lint for errors only
```

Only Black is currently enforced in CI (isort and pylint checks are commented out in `.github/workflows/lint.yml`).

### Documentation
```bash
poetry run pip install -r docs/requirements.txt
poetry run mkdocs serve                  # local dev server at localhost:8000
```

## Architecture

### Core Tracking Pipeline (`norfair/`)

- **`tracker.py`** — `Tracker` class (main entry point), `Detection` (input), `TrackedObject` (output). Orchestrates the tracking loop: predict → match detections via distance function → update Kalman filters.
- **`distances.py`** — Distance function abstractions (`Distance`, `ScalarDistance` ABCs) and built-in functions (Euclidean, IoU, etc.). Uses scipy for spatial computations. Custom distance functions must follow these interfaces.
- **`filter.py`** — Kalman filter implementations behind `FilterFactory` ABC. Three implementations: `OptimizedKalmanFilterFactory` (default, custom), `FilterPyKalmanFilterFactory` (wraps filterpy), `NoFilterFactory`.
- **`camera_motion.py`** — Camera motion estimation and coordinate transformation for moving-camera scenarios. Requires OpenCV.
- **`video.py`** — `Video` class for input/output video handling with OpenCV. Provides iterator interface over frames with progress bars (rich).
- **`metrics.py`** — MOT Challenge metrics evaluation. Requires optional `motmetrics` dependency.
- **`drawing/`** — Visualization subpackage for rendering tracked objects, paths, bounding boxes, and grids onto frames.

### Key Design Patterns

- **ABC + Factory pattern**: `FilterFactory` and `Distance` are abstract bases allowing users to plug in custom implementations.
- **Optional dependencies**: OpenCV and motmetrics are optional extras (`video`, `metrics`). Modules using them handle import errors gracefully.
- **NumPy compatibility**: Supports both NumPy 1.23+ and 2.x. The MOT metrics test environment pins NumPy < 2 due to motmetrics incompatibility.

### Testing Structure (`tests/`)

- `conftest.py` — Fixtures for creating mock detections and tracked objects
- `test_tracker.py` — Core tracking logic tests
- `test_distances.py` — Distance function tests
- `test_drawing.py` — Drawing functionality tests
- `mot_metrics.py` — Integration test that downloads MOT17 dataset (run separately via tox `mot-py313` env)

## Code Style

- Black formatter, line length 88
- isort with Black profile
- NumPy-style docstrings
- Type hints throughout (includes `py.typed` marker)
