import numpy as np
import pytest

from norfair import (
    Detection,
    FilterPyKalmanFilterFactory,
    OptimizedKalmanFilterFactory,
    Tracker,
)
from norfair.utils import validate_points


def test_params():
    #
    # test some invalid initializations
    #
    with pytest.raises(ValueError):
        Tracker("euclidean", distance_threshold=10, initialization_delay=-1)
    with pytest.raises(ValueError):
        Tracker(
            "euclidean",
            distance_threshold=10,
            initialization_delay=1,
            hit_counter_max=0,
        )
    with pytest.raises(ValueError):
        Tracker(
            "euclidean",
            distance_threshold=10,
            initialization_delay=1,
            hit_counter_max=1,
        )
    with pytest.raises(ValueError):
        Tracker(
            "_bad_distance",
            distance_threshold=10,
            initialization_delay=1,
            hit_counter_max=1,
        )


@pytest.mark.parametrize(
    "filter_factory", [FilterPyKalmanFilterFactory(), OptimizedKalmanFilterFactory()]
)
def test_simple(filter_factory):
    for delay in [0, 1, 3]:
        for counter_max in [delay + 1, delay + 3]:
            #
            # tests a simple static detection
            #
            tracker = Tracker(
                "euclidean",
                initialization_delay=delay,
                distance_threshold=100,
                hit_counter_max=counter_max,
                filter_factory=filter_factory,
            )

            detections = [Detection(points=np.array([[1, 1]]))]

            # test the delay
            for _age in range(delay):
                assert len(tracker.update(detections)) == 0

            # build up hit_counter from delay+1 to counter_max
            for age in range(delay, counter_max):
                tracked_objects = tracker.update(detections)
                assert len(tracked_objects) == 1
                obj = tracked_objects[0]
                np.testing.assert_almost_equal(
                    tracked_objects[0].estimate, np.array([[1, 1]])
                )
                assert obj.age == age
                assert obj.hit_counter == age + 1

            # check that counter is capped at counter_max
            for age in range(counter_max, counter_max + 3):
                tracked_objects = tracker.update(detections)
                assert len(tracked_objects) == 1
                obj = tracked_objects[0]
                np.testing.assert_almost_equal(
                    tracked_objects[0].estimate, np.array([[1, 1]])
                )
                assert obj.age == age
                assert obj.hit_counter == counter_max

            # check that counter goes down to 0 wen no detections
            # Set age explicitly after previous loop (was counter_max + 2)
            age = counter_max + 2
            for counter in range(counter_max - 1, -1, -1):
                age += 1
                tracked_objects = tracker.update()
                assert len(tracked_objects) == 1
                obj = tracked_objects[0]
                np.testing.assert_almost_equal(
                    tracked_objects[0].estimate, np.array([[1, 1]])
                )
                assert obj.age == age
                assert obj.hit_counter == counter

            # check that object dissapears in the next frame
            assert len(tracker.update()) == 0


@pytest.mark.parametrize(
    "filter_factory", [FilterPyKalmanFilterFactory(), OptimizedKalmanFilterFactory()]
)
def test_moving(filter_factory):
    #
    # Test a simple case of a moving object
    #
    tracker = Tracker(
        "euclidean",
        initialization_delay=3,
        distance_threshold=100,
        filter_factory=filter_factory,
    )

    assert len(tracker.update([Detection(points=np.array([[1, 1]]))])) == 0
    assert len(tracker.update([Detection(points=np.array([[1, 2]]))])) == 0
    assert len(tracker.update([Detection(points=np.array([[1, 3]]))])) == 0
    tracked_objects = tracker.update([Detection(points=np.array([[1, 4]]))])
    assert len(tracked_objects) == 1

    # check that the estimated position makes sense
    assert tracked_objects[0].estimate[0][0] == 1
    assert 3 < tracked_objects[0].estimate[0][1] <= 4


@pytest.mark.parametrize(
    "filter_factory", [FilterPyKalmanFilterFactory(), OptimizedKalmanFilterFactory()]
)
def test_distance_t(filter_factory):
    #
    # Test a moving object with a small distance threshold
    #
    tracker = Tracker(
        "euclidean",
        initialization_delay=1,
        distance_threshold=1,
        filter_factory=filter_factory,
    )

    # should not match because the distance is too large
    assert len(tracker.update([Detection(points=np.array([[1, 1]]))])) == 0
    assert len(tracker.update([Detection(points=np.array([[1, 2]]))])) == 0
    assert len(tracker.update([Detection(points=np.array([[1, 3]]))])) == 0
    assert len(tracker.update([Detection(points=np.array([[1, 4]]))])) == 0
    # a closer point should match
    tracked_objects = tracker.update([Detection(points=np.array([[1, 4.1]]))])
    assert len(tracked_objects) == 1

    # check that the estimated position makes sense
    assert tracked_objects[0].estimate[0][0] == 1
    assert 4 < tracked_objects[0].estimate[0][1] <= 4.5


@pytest.mark.parametrize(
    "filter_factory", [FilterPyKalmanFilterFactory(), OptimizedKalmanFilterFactory()]
)
def test_1d_points(filter_factory, mock_coordinate_transformation):
    #
    # Test a detection with rank 1
    #
    tracker = Tracker(
        "euclidean",
        initialization_delay=0,
        distance_threshold=1,
        filter_factory=filter_factory,
    )
    detection = Detection(points=np.array([1, 1]))
    assert detection.points.shape == (1, 2)
    assert detection.absolute_points.shape == (1, 2)
    tracked_objects = tracker.update([detection])
    assert len(tracked_objects) == 1
    tracked_object = tracked_objects[0]
    assert tracked_object.estimate.shape == (1, 2)


def test_camera_motion(mock_coordinate_transformation):
    #
    # Simple test for camera motion
    #
    for one_d in [True, False]:
        tracker = Tracker("euclidean", 1, initialization_delay=0)
        if one_d:
            absolute_points = np.array([1, 1])
        else:
            absolute_points = np.array([[1, 1]])

        relative_points = absolute_points + 1

        coord_transformation_mock = mock_coordinate_transformation(
            relative_points=relative_points, absolute_points=absolute_points
        )

        detection = Detection(relative_points)
        tracked_objects = tracker.update(
            [detection], coord_transformations=coord_transformation_mock
        )

        # assert that the detection was correctly updated
        np.testing.assert_equal(
            detection.absolute_points, validate_points(absolute_points)
        )
        np.testing.assert_equal(detection.points, validate_points(relative_points))

        # check the tracked_object
        assert len(tracked_objects) == 1
        obj = tracked_objects[0]
        np.testing.assert_almost_equal(
            obj.get_estimate(absolute=False), validate_points(relative_points)
        )
        np.testing.assert_almost_equal(
            obj.get_estimate(absolute=True), validate_points(absolute_points)
        )
        np.testing.assert_almost_equal(obj.estimate, validate_points(relative_points))


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_count(delay):
    #
    # Test trackers count of objects
    #
    for counter_max in [delay + 1, delay + 3]:
        #
        # tests a simple static detection
        #
        tracker = Tracker(
            "euclidean",
            initialization_delay=delay,
            distance_threshold=1,
            hit_counter_max=counter_max,
        )

        detections = [Detection(points=np.array([[1, 1]]))]
        for _ in range(delay):
            assert len(tracker.update(detections)) == 0
            assert tracker.total_object_count == 0
            assert tracker.current_object_count == 0

        assert len(tracker.update(detections)) == 1
        assert tracker.total_object_count == 1
        assert tracker.current_object_count == 1

        for _ in range(delay + 1, 0, -1):
            assert len(tracker.update()) == 1
            assert tracker.total_object_count == 1
            assert tracker.current_object_count == 1

        assert len(tracker.update()) == 0
        assert tracker.total_object_count == 1
        assert tracker.current_object_count == 0

        detections = [
            Detection(points=np.array([[2, 2]])),
            Detection(points=np.array([[3, 3]])),
        ]
        # test the delay
        for _ in range(delay):
            assert len(tracker.update(detections)) == 0
            assert tracker.total_object_count == 1
            assert tracker.current_object_count == 0

        assert len(tracker.update(detections)) == 2
        assert tracker.total_object_count == 3
        assert tracker.current_object_count == 2

        for _ in range(delay + 1, 0, -1):
            assert len(tracker.update()) == 2
            assert tracker.total_object_count == 3
            assert tracker.current_object_count == 2

        assert len(tracker.update()) == 0
        assert tracker.total_object_count == 3
        assert tracker.current_object_count == 0


def test_multiple_trackers():
    tracker1 = Tracker(
        "euclidean",
        initialization_delay=0,
        distance_threshold=1,
        hit_counter_max=2,
    )
    tracker2 = Tracker(
        "euclidean",
        initialization_delay=0,
        distance_threshold=1,
        hit_counter_max=2,
    )
    detections1 = [Detection(points=np.array([[1, 1]]))]
    detections2 = [Detection(points=np.array([[2, 2]]))]

    tracked_objects1 = tracker1.update(detections1)
    assert len(tracked_objects1) == 1
    tracked_objects2 = tracker2.update(detections2)
    assert len(tracked_objects2) == 1

    assert tracker1.total_object_count == 1
    assert tracker2.total_object_count == 1


def test_reid_hit_counter():
    #
    # test reid hit counter and initializations
    #

    # simple reid distance
    def dist(new_obj, tracked_obj):
        return float(np.linalg.norm(new_obj.estimate - tracked_obj.estimate))

    hit_counter_max = 2
    reid_hit_counter_max = 2

    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=1,
        hit_counter_max=hit_counter_max,
        initialization_delay=1,
        reid_distance_function=dist,
        reid_distance_threshold=5,
        reid_hit_counter_max=reid_hit_counter_max,
    )

    tracked_objects = tracker.update([Detection(points=np.array([[1, 1]]))])
    tracked_objects = tracker.update([Detection(points=np.array([[1, 1]]))])

    # check that hit counters initialize correctly
    assert len(tracked_objects) == 1
    assert tracked_objects[0].hit_counter == 2
    assert tracked_objects[0].reid_hit_counter is None

    # check that object is dead if it doesn't get matched to any detections
    obj_id = tracked_objects[0].id
    for _ in range(hit_counter_max + 1):
        tracked_objects = tracker.update()
    assert len(tracked_objects) == 0

    # check that previous object gets back to life after reid matching
    for _ in range(hit_counter_max):
        tracked_objects = tracker.update([Detection(points=np.array([[2, 2]]))])
    assert len(tracked_objects) == 1
    assert tracked_objects[0].id == obj_id
    assert tracked_objects[0].reid_hit_counter is None
    assert tracked_objects[0].hit_counter == hit_counter_max

    # check that previous object gets eliminated after hit_counter_max + reid_hit_counter_max + 1
    for _ in range(hit_counter_max + reid_hit_counter_max + 1):
        tracked_objects = tracker.update()
    assert len(tracked_objects) == 0
    for _ in range(2):
        tracked_objects = tracker.update([Detection(points=np.array([[1, 1]]))])
    assert len(tracked_objects) == 1
    assert tracked_objects[0].id != obj_id


def test_reid_hit_counter_reset():
    #
    # test that reid hit counter resets to None if it had started counting down but
    # then the track was hit with an incoming detection
    #

    # simple reid distance
    def dist(new_obj, tracked_obj):
        return float(np.linalg.norm(new_obj.estimate - tracked_obj.estimate))

    hit_counter_max = 2
    reid_hit_counter_max = 2

    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=1,
        hit_counter_max=hit_counter_max,
        initialization_delay=1,
        reid_distance_function=dist,
        reid_distance_threshold=5,
        reid_hit_counter_max=reid_hit_counter_max,
    )

    # check that hit counters initialize correctly
    tracked_objects = tracker.update([Detection(points=np.array([[1, 1]]))])
    tracked_objects = tracker.update([Detection(points=np.array([[1, 1]]))])
    assert len(tracked_objects) == 1
    assert tracked_objects[0].hit_counter == 2
    assert tracked_objects[0].reid_hit_counter is None

    # check that object is still alive when hit_counter goes to 0
    obj_id = tracked_objects[0].id
    for _ in range(hit_counter_max):
        tracked_objects = tracker.update()
    assert len(tracked_objects) == 1
    assert tracked_objects[0].hit_counter == 0
    assert tracked_objects[0].reid_hit_counter is None

    # check that object is alive and reid_hit_counter is None after being matched again
    tracked_objects = tracker.update([Detection(points=np.array([[1, 1]]))])
    assert len(tracked_objects) == 1
    assert tracked_objects[0].hit_counter == 1
    assert tracked_objects[0].reid_hit_counter is None

    # check that after reid_hit_counter_max more updates, object still exists
    for _ in range(reid_hit_counter_max + 2):
        tracked_objects = tracker.update([Detection(points=np.array([[1, 1]]))])
    assert len(tracked_objects) == 1
    assert tracked_objects[0].hit_counter == 2
    assert tracked_objects[0].reid_hit_counter is None
    assert tracked_objects[0].id == obj_id


def test_detection_age_always_set():
    """Test that detection.age is set even when past_detections_length=0."""
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=100,
        hit_counter_max=4,
        initialization_delay=0,
        past_detections_length=0,
    )

    tracked = tracker.update([Detection(points=np.array([[1, 1]]))])
    assert len(tracked) == 1
    # last_detection.age should be set even though past_detections_length=0
    assert tracked[0].last_detection.age is not None
    assert tracked[0].last_detection.age == tracked[0].age

    # Feed more frames and check age stays consistent
    tracked = tracker.update([Detection(points=np.array([[1, 1]]))])
    assert tracked[0].last_detection.age is not None
    assert tracked[0].last_detection.age == tracked[0].age


def test_detection_age_set_when_buffer_full():
    """Test that detection.age is set even when past_detections buffer is full and not replaced."""
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=100,
        hit_counter_max=10,
        initialization_delay=0,
        past_detections_length=2,
    )

    # Fill the past_detections buffer
    tracked = tracker.update([Detection(points=np.array([[1, 1]]))])
    tracked = tracker.update([Detection(points=np.array([[1, 1]]))])
    tracked = tracker.update([Detection(points=np.array([[1, 1]]))])
    assert len(tracked) == 1

    # At this point buffer may be full; the detection's age should still be set
    assert tracked[0].last_detection.age is not None
    assert tracked[0].last_detection.age == tracked[0].age


def test_detection_age_after_merge():
    """Test that detection ages are updated correctly after merging tracked objects."""

    def reid_dist(new_obj, tracked_obj):
        return float(np.linalg.norm(new_obj.estimate - tracked_obj.estimate))

    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=1,
        hit_counter_max=4,
        initialization_delay=1,
        reid_distance_function=reid_dist,
        reid_distance_threshold=10,
        reid_hit_counter_max=5,
        past_detections_length=4,
    )

    # Create first track (will become initialized)
    tracker.update([Detection(points=np.array([[1, 1]]))])
    tracker.update([Detection(points=np.array([[1, 1]]))])
    tracked = tracker.update([Detection(points=np.array([[1, 1]]))])
    assert len(tracked) == 1
    obj_id = tracked[0].id

    # Let the object die (hit_counter goes negative)
    for _ in range(5):
        tracker.update()

    # Create a new detection nearby that will trigger re-id merge
    tracker.update([Detection(points=np.array([[2, 2]]))])
    tracked = tracker.update([Detection(points=np.array([[2, 2]]))])
    assert len(tracked) == 1
    assert tracked[0].id == obj_id

    # Verify that last_detection.age is set to the current object age
    assert tracked[0].last_detection.age is not None
    assert tracked[0].last_detection.age == tracked[0].age

    # Verify past detections have ages set (not None)
    for pd in tracked[0].past_detections:
        assert pd.age is not None


def test_label_matching():
    """Test that detections with different labels are not matched together."""
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=100,
        hit_counter_max=4,
        initialization_delay=0,
    )

    # Two detections at same point but different labels
    det_a = Detection(points=np.array([[1, 1]]), label="car")
    det_b = Detection(points=np.array([[1, 1]]), label="person")
    tracked = tracker.update([det_a, det_b])
    assert len(tracked) == 2

    # Each should maintain its label
    labels = {obj.label for obj in tracked}
    assert labels == {"car", "person"}


def test_nan_distance_raises():
    """Test that NaN in distance matrix raises ValueError."""

    def bad_distance(det, obj):
        return float("nan")

    tracker = Tracker(
        distance_function=bad_distance,
        distance_threshold=100,
        hit_counter_max=4,
        initialization_delay=0,
    )

    tracker.update([Detection(points=np.array([[1, 1]]))])
    with pytest.raises(ValueError, match="nan"):
        tracker.update([Detection(points=np.array([[1, 1]]))])


def test_period_parameter():
    """Test that the period parameter affects hit counter increments."""
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=100,
        hit_counter_max=20,
        initialization_delay=0,
    )

    det = [Detection(points=np.array([[1, 1]]))]
    tracked = tracker.update(det, period=5)
    assert len(tracked) == 1
    # With period=5, hit_counter should be 5 (initial period)
    assert tracked[0].hit_counter == 5


def test_estimate_velocity():
    """Test that estimate_velocity returns expected shape."""
    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=100,
        hit_counter_max=4,
        initialization_delay=0,
    )

    tracked = tracker.update([Detection(points=np.array([[1, 1]]))])
    assert len(tracked) == 1
    vel = tracked[0].estimate_velocity
    assert vel.shape == (1, 2)


def test_global_id():
    """Test that global_id is assigned and unique across trackers."""
    tracker1 = Tracker(
        distance_function="euclidean",
        distance_threshold=100,
        initialization_delay=0,
    )
    tracker2 = Tracker(
        distance_function="euclidean",
        distance_threshold=100,
        initialization_delay=0,
    )

    t1 = tracker1.update([Detection(points=np.array([[1, 1]]))])
    t2 = tracker2.update([Detection(points=np.array([[2, 2]]))])

    assert t1[0].global_id is not None
    assert t2[0].global_id is not None
    assert t1[0].global_id != t2[0].global_id


def test_global_id_thread_safety():
    """Test that global_id counter is thread-safe."""
    import threading

    from norfair.tracker import _TrackedObjectFactory

    # Reset counter
    _TrackedObjectFactory.global_count = 0

    results = []

    def create_objects(num_objects):
        factory = _TrackedObjectFactory()
        for _ in range(num_objects):
            local_id, global_id = factory.get_ids()
            results.append(global_id)

    # Create multiple threads that create objects concurrently
    threads = []
    num_threads = 10
    objects_per_thread = 10

    for _ in range(num_threads):
        thread = threading.Thread(target=create_objects, args=(objects_per_thread,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Verify all global_ids are unique
    assert len(results) == num_threads * objects_per_thread
    assert len(set(results)) == len(results), "Duplicate global_ids detected"


def test_detection_scores_broadcast():
    """Test that scalar scores are broadcast to all points."""
    import numpy as np

    from norfair import Detection

    # Test with integer score
    det = Detection(points=np.array([[1, 2], [3, 4]]), scores=5)
    assert det.scores is not None
    assert len(det.scores) == 2
    assert det.scores[0] == 5
    assert det.scores[1] == 5

    # Test with float score
    det = Detection(points=np.array([[1, 2], [3, 4]]), scores=0.75)
    assert det.scores is not None
    assert len(det.scores) == 2
    assert det.scores[0] == 0.75
    assert det.scores[1] == 0.75


def test_tracked_object_scores_attribute():
    """Test that TrackedObject.scores is set from matched detection."""
    import numpy as np

    from norfair import Detection, Tracker

    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=100,
        initialization_delay=0,
    )

    # Create detection with scores
    det = Detection(points=np.array([[1, 1]]), scores=np.array([0.9]))
    tracked = tracker.update([det])

    assert len(tracked) == 1
    assert tracked[0].scores is not None
    assert tracked[0].scores[0] == 0.9

    # After tracker_step without match, scores should be None
    tracker.update()
    assert tracked[0].scores is None


def test_empty_candidates_and_objects():
    """Test distance functions with empty candidates or objects."""
    from norfair.distances import ScalarDistance, VectorizedDistance, frobenius

    # Test ScalarDistance with empty lists
    scalar_dist = ScalarDistance(frobenius)
    dist_matrix = scalar_dist.get_distances([], [])
    assert dist_matrix.shape == (0, 0)

    # Test VectorizedDistance with empty lists
    def vec_dist_func(cands, objs):
        return np.zeros((len(cands), len(objs)))

    vec_dist = VectorizedDistance(vec_dist_func)
    dist_matrix = vec_dist.get_distances([], [])
    assert dist_matrix.shape == (0, 0)


def test_isinstance_type_checking_in_vectorized_distance(mock_det, mock_obj):
    """Test that VectorizedDistance uses isinstance for type checking."""
    from norfair.distances import VectorizedDistance
    from norfair.tracker import Detection

    def dist_func(cands, objs):
        return np.zeros((len(cands), len(objs)))

    vd = VectorizedDistance(dist_func)

    # Create a Detection and a TrackedObject
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])

    # This should work with isinstance checking
    dist_matrix = vd.get_distances([obj], [det])
    assert dist_matrix.shape == (1, 1)


def test_iou_validates_both_candidates_and_objects():
    """Test that iou validates both candidates and objects bounding boxes."""
    from norfair.distances import iou

    # Valid candidates, invalid objects
    valid_candidates = np.array([[0, 0, 2, 2]])
    invalid_objects = np.array([[0, 0]])  # Wrong shape

    with pytest.raises(ValueError, match="must be defined as np.array with"):
        iou(valid_candidates, invalid_objects)

    # Invalid candidates, valid objects
    invalid_candidates = np.array([[0, 0]])  # Wrong shape
    valid_objects = np.array([[0, 0, 2, 2]])

    with pytest.raises(ValueError, match="must be defined as np.array with"):
        iou(invalid_candidates, valid_objects)


def test_detection_with_single_float_score():
    """Test Detection accepts single float/int score (not just np.ndarray)."""
    import numpy as np

    from norfair import Detection

    # Single point with float score
    det = Detection(points=np.array([[1, 2]]), scores=0.8)
    assert det.scores is not None
    assert isinstance(det.scores, np.ndarray)
    assert len(det.scores) == 1
    assert det.scores[0] == 0.8

    # Multiple points with single float score (should broadcast)
    det = Detection(points=np.array([[1, 2], [3, 4], [5, 6]]), scores=0.9)
    assert det.scores is not None
    assert len(det.scores) == 3
    np.testing.assert_array_equal(det.scores, [0.9, 0.9, 0.9])