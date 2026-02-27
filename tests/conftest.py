import numpy as np
import pytest

from norfair.tracker import Detection, _TrackedObjectFactory
from norfair.utils import validate_points


@pytest.fixture(autouse=True)
def reset_global_count():
    """Reset TrackedObject global count before each test for isolation."""
    _TrackedObjectFactory.global_count = 0
    yield


@pytest.fixture
def mock_det():
    def _make_detection(points, scores=None, label=None):
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        return Detection(points=points, scores=scores, label=label)

    return _make_detection


@pytest.fixture
def mock_obj(mock_det):
    class FakeTrackedObject:
        def __init__(self, points, scores=None, label=None):
            if not isinstance(points, np.ndarray):
                points = np.array(points)

            self.estimate = validate_points(points)
            self.last_detection = mock_det(points, scores=scores)
            self.label = label

    return FakeTrackedObject


@pytest.fixture
def mock_coordinate_transformation():
    # simple mock to return abs or relative positions
    class TransformMock:
        def __init__(self, relative_points, absolute_points) -> None:
            self.absolute_points = validate_points(absolute_points)
            self.relative_points = validate_points(relative_points)

        def abs_to_rel(self, points):
            np.testing.assert_equal(points, self.absolute_points)
            return self.relative_points

        def rel_to_abs(self, points):
            np.testing.assert_equal(points, self.relative_points)
            return self.absolute_points

    return TransformMock
