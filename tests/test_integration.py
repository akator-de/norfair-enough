"""End-to-end integration tests for the tracking pipeline.

These scenarios drive a ``Tracker`` through multi-frame synthetic
detection streams and assert on library-level outputs (track IDs,
estimate positions, track counts). They exercise the full
predict → match → update loop that individual unit tests don't cover
end-to-end, and are fully deterministic with no external data.
"""

import numpy as np

from norfair import Detection, Tracker
from norfair.camera_motion import TranslationTransformation


def _detections(point_arrays):
    """Build a ``Detection`` list from an iterable of point arrays."""
    return [
        Detection(points=np.asarray(p, dtype=float).reshape(-1, 2))
        for p in point_arrays
    ]


def test_linear_motion_single_track_is_stable():
    """One linearly-moving object stays on a single stable track ID."""
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=20.0,
        initialization_delay=2,
        hit_counter_max=15,
    )

    seen_ids: set[int] = set()
    for frame in range(30):
        x = 10.0 + frame * 2.0
        tracked = tracker.update(_detections([[[x, 50.0]]]))
        for obj in tracked:
            assert obj.id is not None
            seen_ids.add(obj.id)
            assert obj.estimate.shape == (1, 2)

    assert len(seen_ids) == 1


def test_two_well_separated_tracks_keep_distinct_ids():
    """Two parallel trajectories with large separation produce two stable IDs.

    The IDs assigned at the first post-initialization frame must remain
    pinned to the same y-position on every subsequent frame — catching
    transient swaps, not just the final state.
    """
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=20.0,
        initialization_delay=2,
    )

    top_id: int | None = None
    bottom_id: int | None = None

    tracked: list = []
    for frame in range(25):
        x = 10.0 + frame * 2.0
        tracked = tracker.update(_detections([[[x, 50.0]], [[x, 250.0]]]))
        if len(tracked) != 2:
            continue
        by_y = {
            ("top" if obj.estimate[0, 1] < 150 else "bottom"): obj.id for obj in tracked
        }
        if top_id is None:
            top_id, bottom_id = by_y["top"], by_y["bottom"]
            assert top_id != bottom_id
        else:
            assert by_y["top"] == top_id, f"top track ID swapped at frame {frame}"
            assert by_y["bottom"] == bottom_id, (
                f"bottom track ID swapped at frame {frame}"
            )

    assert top_id is not None and bottom_id is not None

    y_values = sorted(obj.estimate[0, 1] for obj in tracked)
    assert abs(y_values[0] - 50.0) < 5.0
    assert abs(y_values[1] - 250.0) < 5.0


def test_short_occlusion_preserves_id():
    """A brief gap in detections (within hit-counter budget) keeps the same ID."""
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=30.0,
        initialization_delay=2,
        hit_counter_max=15,
    )

    established_id = None
    for frame in range(10):
        x = 10.0 + frame * 2.0
        tracked = tracker.update(_detections([[[x, 50.0]]]))
        if tracked:
            established_id = tracked[0].id
    assert established_id is not None

    # drop detections for several frames, well within the hit-counter budget
    for _ in range(4):
        tracker.update([])

    # resume and verify the ID is reused
    reused = False
    for frame in range(14, 22):
        x = 10.0 + frame * 2.0
        tracked = tracker.update(_detections([[[x, 50.0]]]))
        if any(obj.id == established_id for obj in tracked):
            reused = True
    assert reused


def test_long_occlusion_terminates_track():
    """Once a track misses more frames than the hit-counter allows, it disappears."""
    hit_counter_max = 6
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=30.0,
        initialization_delay=2,
        hit_counter_max=hit_counter_max,
    )

    for frame in range(10):
        tracker.update(_detections([[[10.0 + frame, 50.0]]]))

    for _ in range(hit_counter_max + 2):
        tracker.update([])

    assert tracker.update([]) == []


def test_initialization_delay_hides_track_until_confirmed():
    """A new track is withheld until hit_counter exceeds initialization_delay."""
    initialization_delay = 3
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=30.0,
        initialization_delay=initialization_delay,
    )

    for _ in range(initialization_delay):
        assert tracker.update(_detections([[[1.0, 1.0]]])) == []

    tracked = tracker.update(_detections([[[1.0, 1.0]]]))
    assert len(tracked) == 1


def test_camera_translation_compensation_keeps_world_position_stable():
    """Object fixed in world space, camera translating — the absolute-frame
    estimate must stay on the world point when coord_transformations is
    supplied every frame."""
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=5.0,
        initialization_delay=2,
        hit_counter_max=15,
    )

    world_point = np.array([[100.0, 100.0]])
    camera_velocity = np.array([1.5, 0.5])

    tracked: list = []
    for frame in range(20):
        camera_offset = camera_velocity * frame
        apparent = world_point - camera_offset
        # TranslationTransformation: abs_to_rel(p) = p + movement_vector.
        # We need world + movement_vector == apparent, so movement_vector
        # equals -camera_offset.
        transform = TranslationTransformation(-camera_offset)

        tracked = tracker.update(
            [Detection(points=apparent.copy())],
            coord_transformations=transform,
        )

    assert len(tracked) == 1
    np.testing.assert_allclose(
        tracked[0].get_estimate(absolute=True), world_point, atol=1.0
    )
