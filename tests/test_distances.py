import numpy as np
import pytest

from norfair.distances import (
    ScalarDistance,
    ScipyDistance,
    VectorizedDistance,
    create_keypoints_voting_distance,
    create_normalized_mean_euclidean_distance,
    frobenius,
    get_distance_by_name,
)


def test_frobenius(mock_obj, mock_det):
    fro = get_distance_by_name("frobenius")

    # perfect match
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])
    np.testing.assert_almost_equal(fro.distance_function(det, obj), 0)

    # foat type
    det = mock_det([[1.1, 2.2], [3.3, 4.4]])
    obj = mock_obj([[1.1, 2.2], [3.3, 4.4]])
    np.testing.assert_almost_equal(fro.distance_function(det, obj), 0)

    # distance of 1 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 2], [3, 4]])
    np.testing.assert_almost_equal(fro.distance_function(det, obj), np.sqrt(1))

    # distance of 2 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [3, 4]])
    np.testing.assert_almost_equal(fro.distance_function(det, obj), 2)

    # distance of 1 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 3], [4, 5]])
    np.testing.assert_almost_equal(fro.distance_function(det, obj), np.sqrt(4))

    # negative difference
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(fro.distance_function(det, obj), 2)

    # negative equals
    det = mock_det([[-1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(fro.distance_function(det, obj), 0)


def test_mean_manhattan(mock_det, mock_obj):
    man = get_distance_by_name("mean_manhattan")

    # perfect match
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])
    np.testing.assert_almost_equal(man.distance_function(det, obj), 0)

    # foat type
    det = mock_det([[1.1, 2.2], [3.3, 4.4]])
    obj = mock_obj([[1.1, 2.2], [3.3, 4.4]])
    np.testing.assert_almost_equal(man.distance_function(det, obj), 0)

    # distance of 1 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 2], [3, 4]])
    np.testing.assert_almost_equal(man.distance_function(det, obj), 1 / 2)

    # distance of 2 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [3, 4]])
    np.testing.assert_almost_equal(man.distance_function(det, obj), 1)

    # distance of 1 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 3], [4, 5]])
    np.testing.assert_almost_equal(man.distance_function(det, obj), 2)

    # negative difference
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(man.distance_function(det, obj), 1)

    # negative equals
    det = mock_det([[-1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(man.distance_function(det, obj), 0)


def test_mean_euclidean(mock_det, mock_obj):
    euc = get_distance_by_name("mean_euclidean")

    # perfect match
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), 0)

    # foat type
    det = mock_det([[1.1, 2.2], [3.3, 4.4]])
    obj = mock_obj([[1.1, 2.2], [3.3, 4.4]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), 0)

    # distance of 1 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 2], [3, 4]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), 1 / 2)

    # distance of 2 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [3, 4]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), 1)

    # distance of 2 in 1 dimension of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [5, 4]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), 2)

    # distance of 2 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 4], [5, 6]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), np.sqrt(8))

    # distance of 1 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 3], [4, 5]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), np.sqrt(2))

    # negative difference
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), 1)

    # negative equals
    det = mock_det([[-1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(euc.distance_function(det, obj), 0)


def test_iou():
    iou = get_distance_by_name("iou")

    # perfect match
    det = np.array([[0, 0, 1, 1]])
    obj = np.array([[0, 0, 1, 1]])
    np.testing.assert_almost_equal(iou.distance_function(det, obj), 0)

    # float type
    det = np.array([[0.0, 0.0, 1.1, 1.1]])
    obj = np.array([[0.0, 0.0, 1.1, 1.1]])
    np.testing.assert_almost_equal(iou.distance_function(det, obj), 0)

    # det contained in obj
    det = np.array([[0, 0, 1, 1]])
    obj = np.array([[0, 0, 2, 2]])
    np.testing.assert_almost_equal(iou.distance_function(det, obj), 1 - 1 / 4)

    # no overlap
    det = np.array([[0, 0, 1, 1]])
    obj = np.array([[1, 1, 2, 2]])
    np.testing.assert_almost_equal(iou.distance_function(det, obj), 1)

    # obj fully contained on det
    det = np.array([[0, 0, 4, 4]])
    obj = np.array([[1, 1, 2, 2]])
    np.testing.assert_almost_equal(iou.distance_function(det, obj), 1 - 1 / 16)

    # partial overlap
    det = np.array([[0, 0, 2, 2]])
    obj = np.array([[1, 1, 3, 3]])
    np.testing.assert_almost_equal(iou.distance_function(det, obj), 1 - 1 / (8 - 1))

    # invalid bbox
    det = np.array([[0, 0]])
    obj = np.array([[0, 0]])
    with pytest.raises(ValueError):
        iou.distance_function(det, obj)

    # invalid bbox
    det = np.array([[0, 0, 1, 1, 2, 2]])
    obj = np.array([[0, 0, 2, 2]])
    with pytest.raises(ValueError):
        iou.distance_function(det, obj)


def test_keypoint_vote(mock_obj, mock_det):
    vote_d = create_keypoints_voting_distance(
        keypoint_distance_threshold=np.sqrt(8), detection_threshold=0.5
    )

    # perfect match
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=0.6)
    obj = mock_obj(points=[[0, 0], [1, 1], [2, 2]], scores=0.6)
    np.testing.assert_almost_equal(vote_d(det, obj), 1 / 4)  # 3 matches

    # just under distance threshold
    det = mock_det(points=[[0, 0], [1, 1], [2, 2.0]], scores=0.6)
    obj = mock_obj(points=[[0, 0], [1, 1], [4, 3.9]], scores=0.6)
    np.testing.assert_almost_equal(vote_d(det, obj), 1 / 4)  # 3 matches

    # just above distance threshold
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=0.6)
    obj = mock_obj(points=[[0, 0], [1, 1], [4, 4]], scores=0.6)
    np.testing.assert_almost_equal(vote_d(det, obj), 1 / 3)  # 2 matches

    # just under score threshold on detection
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=[0.6, 0.6, 0.5])
    obj = mock_obj(points=[[0, 0], [1, 1], [2, 2]], scores=[0.6, 0.6, 0.6])
    np.testing.assert_almost_equal(vote_d(det, obj), 1 / 3)  # 2 matches

    # just under score threshold on tracked_object's last detection
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=[0.6, 0.6, 0.6])
    obj = mock_obj(points=[[0, 0], [1, 1], [2, 2]], scores=[0.6, 0.6, 0.5])
    np.testing.assert_almost_equal(vote_d(det, obj), 1 / 3)  # 2 matches

    # no match because of scores
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=0.5)
    obj = mock_obj(points=[[0, 0], [1, 1], [2, 2]], scores=0.5)
    np.testing.assert_almost_equal(vote_d(det, obj), 1)  # 0 matches

    # no match because of distances
    det = mock_det(points=[[0, 0], [1, 1], [2, 2]], scores=0.6)
    obj = mock_obj(points=[[2, 2], [3, 3], [4, 4]], scores=0.6)
    np.testing.assert_almost_equal(vote_d(det, obj), 1)  # 0 matches


def test_normalized_euclidean(mock_obj, mock_det):
    norm_e = create_normalized_mean_euclidean_distance(10, 10)

    # perfect match
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0)

    # foat type
    det = mock_det([[1.1, 2.2], [3.3, 4.4]])
    obj = mock_obj([[1.1, 2.2], [3.3, 4.4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0)

    # distance of 1 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 2], [3, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0.05)

    # distance of 2 in 1 dimension of 1 point
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [3, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0.1)

    # distance of 2 in 1 dimension of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 2], [5, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0.2)

    # distance of 2 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[3, 4], [5, 6]])
    np.testing.assert_almost_equal(norm_e(det, obj), np.sqrt(8) / 10)

    # distance of 1 in all dimensions of all points
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[2, 3], [4, 5]])
    np.testing.assert_almost_equal(norm_e(det, obj), np.sqrt(2) / 10)

    # negative difference
    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0.1)

    # negative equals
    det = mock_det([[-1, 2], [3, 4]])
    obj = mock_obj([[-1, 2], [3, 4]])
    np.testing.assert_almost_equal(norm_e(det, obj), 0)


def test_scalar_distance(mock_obj, mock_det):
    fro = ScalarDistance(frobenius)

    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])

    dist_matrix = fro.get_distances([obj], [det])

    assert isinstance(dist_matrix, np.ndarray)
    assert dist_matrix.shape == (1, 1)
    assert dist_matrix[0, 0] == 0


def test_vectorized_distance(mock_obj, mock_det):
    def distance_function(cands, objs):
        distance_matrix = np.full(
            (len(cands), len(objs)),
            fill_value=np.inf,
            dtype=np.float32,
        )
        for c, cand in enumerate(cands):
            for o, obj in enumerate(objs):
                distance_matrix[c, o] = np.linalg.norm(cand - obj)
        return distance_matrix

    fro = VectorizedDistance(distance_function)

    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [3, 4]])

    dist_matrix = fro.get_distances([obj], [det])

    assert isinstance(dist_matrix, np.ndarray)
    assert dist_matrix.shape == (1, 1)
    assert dist_matrix[0, 0] == 0


def test_scipy_distance(mock_obj, mock_det):
    euc = ScipyDistance("euclidean")

    det = mock_det([[1, 2], [3, 4]])
    obj = mock_obj([[1, 2], [4, 4]])

    dist_matrix = euc.get_distances([obj], [det])

    assert isinstance(dist_matrix, np.ndarray)
    assert dist_matrix.shape == (1, 1)
    assert dist_matrix[0, 0] == 1.0


def test_vectorized_distance_with_labels(mock_obj, mock_det):
    """Test that VectorizedDistance handles labels correctly."""

    def distance_function(cands, objs):
        return np.full(
            (len(cands), len(objs)),
            fill_value=0.5,
            dtype=np.float32,
        )

    vd = VectorizedDistance(distance_function)

    det_car = mock_det([[1, 2], [3, 4]], label="car")
    det_person = mock_det([[5, 6], [7, 8]], label="person")
    obj_car = mock_obj([[1, 2], [3, 4]], label="car")
    obj_person = mock_obj([[5, 6], [7, 8]], label="person")

    dist_matrix = vd.get_distances([obj_car, obj_person], [det_car, det_person])

    assert dist_matrix.shape == (2, 2)
    # Same label should have finite distance
    assert dist_matrix[0, 0] == 0.5  # car-car
    assert dist_matrix[1, 1] == 0.5  # person-person
    # Different labels should be inf
    assert dist_matrix[0, 1] == np.inf  # car-person
    assert dist_matrix[1, 0] == np.inf  # person-car


def test_iou_multi_box():
    """Test IoU with multiple boxes (NxM distance matrix)."""
    from norfair.distances import iou

    candidates = np.array(
        [
            [0, 0, 2, 2],
            [3, 3, 5, 5],
        ]
    )
    objects = np.array(
        [
            [0, 0, 2, 2],
            [1, 1, 3, 3],
            [3, 3, 5, 5],
        ]
    )

    dist = iou(candidates, objects)
    assert dist.shape == (2, 3)
    # Perfect match for first candidate with first object
    np.testing.assert_almost_equal(dist[0, 0], 0.0)
    # Perfect match for second candidate with third object
    np.testing.assert_almost_equal(dist[1, 2], 0.0)
    # No overlap between first candidate and third object
    np.testing.assert_almost_equal(dist[0, 2], 1.0)


def test_scalar_distance_label_mismatch(mock_obj, mock_det):
    """Test that ScalarDistance warns when some detections have labels and some don't."""
    fro = ScalarDistance(frobenius)

    det_with_label = mock_det([[1, 2], [3, 4]], label="car")
    obj_no_label = mock_obj([[1, 2], [3, 4]], label=None)

    dist_matrix = fro.get_distances([obj_no_label], [det_with_label])

    # Should be inf because labels don't match
    assert dist_matrix[0, 0] == np.inf


def test_vectorized_distance_does_not_merge_none_and_str_none(mock_obj, mock_det):
    """VectorizedDistance must treat ``label=None`` and ``label='None'`` as
    distinct labels, not merge them via ``str`` coercion (#97)."""

    def _vec_fn(cands, objs):
        diff = cands[:, None, :] - objs[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    vec = VectorizedDistance(_vec_fn)

    # Objects: one with None, one with the literal string "None".
    obj_none = mock_obj([[1, 2], [3, 4]], label=None)
    obj_str_none = mock_obj([[1, 2], [3, 4]], label="None")

    # Candidates mirror the objects.
    det_none = mock_det([[1, 2], [3, 4]], label=None)
    det_str_none = mock_det([[1, 2], [3, 4]], label="None")

    dist_matrix = vec.get_distances([obj_none, obj_str_none], [det_none, det_str_none])

    # Same-label pairs match (distance 0)
    assert dist_matrix[0, 0] == 0
    assert dist_matrix[1, 1] == 0
    # Cross-label pairs must stay at np.inf — None and "None" are different.
    assert dist_matrix[0, 1] == np.inf, (
        "candidate label=None should not match object label='None'"
    )
    assert dist_matrix[1, 0] == np.inf, (
        "candidate label='None' should not match object label=None"
    )
