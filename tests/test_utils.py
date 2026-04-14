"""Tests for norfair.utils -- comprehensive coverage of all public utilities."""

import logging
from unittest import mock

import numpy as np
import pytest

from norfair.utils import (
    DummyMOTMetricsImport,
    DummyOpenCVImport,
    get_cutout,
    get_terminal_size,
    print_objects_as_table,
    raise_detection_error_message,
    validate_points,
    warn_once,
)


# ---------------------------------------------------------------------------
# validate_points
# ---------------------------------------------------------------------------
class TestValidatePoints:
    """Tests for validate_points: 1D reshape, 2D passthrough, 3D+ rejection."""

    def test_1d_input_reshaped_to_2d(self):
        pts = np.array([10, 20])
        result = validate_points(pts)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result, [[10, 20]])

    def test_1d_single_value(self):
        pts = np.array([5])
        result = validate_points(pts)
        assert result.shape == (1, 1)

    def test_1d_many_values(self):
        pts = np.array([1, 2, 3, 4])
        result = validate_points(pts)
        assert result.shape == (1, 4)

    def test_2d_passthrough(self):
        pts = np.array([[1, 2], [3, 4]])
        result = validate_points(pts)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, pts)

    def test_2d_single_row_passthrough(self):
        pts = np.array([[7, 8]])
        result = validate_points(pts)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result, [[7, 8]])

    def test_3d_raises_value_error(self):
        pts = np.ones((2, 3, 4))
        with pytest.raises(ValueError):
            validate_points(pts)

    def test_4d_raises_value_error(self):
        pts = np.ones((1, 2, 3, 4))
        with pytest.raises(ValueError):
            validate_points(pts)


# ---------------------------------------------------------------------------
# raise_detection_error_message
# ---------------------------------------------------------------------------
class TestRaiseDetectionErrorMessage:
    """Tests for raise_detection_error_message."""

    def test_raises_value_error(self):
        pts = np.ones((2, 3, 4))
        with pytest.raises(ValueError):
            raise_detection_error_message(pts)

    def test_message_contains_shape(self):
        pts = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match=r"\(2, 3, 4\)"):
            raise_detection_error_message(pts)

    def test_message_mentions_detection(self):
        pts = np.ones((5, 5, 5))
        with pytest.raises(ValueError, match="Detection"):
            raise_detection_error_message(pts)

    def test_message_contains_documentation_link(self):
        pts = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="https://"):
            raise_detection_error_message(pts)


# ---------------------------------------------------------------------------
# get_cutout
# ---------------------------------------------------------------------------
class TestGetCutout:
    """Tests for get_cutout input validation, clipping, and cropping."""

    def _make_image(self, h=100, w=200, channels=3):
        return np.arange(h * w * channels, dtype=np.uint8).reshape(h, w, channels)

    def test_normal_cutout_shape(self):
        img = self._make_image()
        points = np.array([[10, 20], [50, 60]])
        cutout = get_cutout(points, img)
        # y: 20..60 = 40 rows, x: 10..50 = 40 cols
        assert cutout.shape == (40, 40, 3)

    def test_normal_cutout_content(self):
        img = self._make_image(h=10, w=10, channels=1)
        points = np.array([[2, 3], [5, 7]])
        cutout = get_cutout(points, img)
        expected = img[3:7, 2:5]
        np.testing.assert_array_equal(cutout, expected)

    def test_single_point_degenerate(self, caplog):
        img = self._make_image()
        points = np.array([[10, 20]])
        with caplog.at_level(logging.WARNING):
            cutout = get_cutout(points, img)
        assert cutout.size == 0

    def test_coords_clipped_to_image_bounds(self):
        img = self._make_image(h=50, w=50)
        points = np.array([[-10, -10], [100, 100]])
        cutout = get_cutout(points, img)
        assert cutout.shape == (50, 50, 3)

    def test_partially_out_of_bounds(self):
        img = self._make_image(h=50, w=50)
        points = np.array([[40, 30], [100, 100]])
        cutout = get_cutout(points, img)
        # x clipped to 40..50 = 10, y clipped to 30..50 = 20
        assert cutout.shape == (20, 10, 3)

    def test_negative_coords_clipped_to_zero(self):
        img = self._make_image(h=50, w=50)
        points = np.array([[-20, -30], [10, 15]])
        cutout = get_cutout(points, img)
        # x: 0..10 = 10, y: 0..15 = 15
        assert cutout.shape == (15, 10, 3)

    def test_degenerate_same_point_warns(self, caplog):
        img = self._make_image()
        points = np.array([[10, 20], [10, 20]])
        with caplog.at_level(logging.WARNING):
            cutout = get_cutout(points, img)
        assert cutout.size == 0
        assert "degenerate" in caplog.text

    def test_degenerate_zero_width_warns(self, caplog):
        img = self._make_image()
        points = np.array([[10, 20], [10, 50]])  # same x
        with caplog.at_level(logging.WARNING):
            cutout = get_cutout(points, img)
        assert cutout.shape[1] == 0  # zero width
        assert "degenerate" in caplog.text

    def test_degenerate_zero_height_warns(self, caplog):
        img = self._make_image()
        points = np.array([[10, 20], [50, 20]])  # same y
        with caplog.at_level(logging.WARNING):
            cutout = get_cutout(points, img)
        assert cutout.shape[0] == 0  # zero height
        assert "degenerate" in caplog.text

    def test_empty_points_raises(self):
        img = self._make_image()
        with pytest.raises(ValueError, match="empty"):
            get_cutout(np.empty((0, 2)), img)

    def test_1d_points_raises(self):
        img = self._make_image()
        with pytest.raises(ValueError, match="shape"):
            get_cutout(np.array([1, 2]), img)

    def test_wrong_columns_raises(self):
        img = self._make_image()
        with pytest.raises(ValueError, match="shape"):
            get_cutout(np.array([[1, 2, 3]]), img)

    def test_3d_points_raises(self):
        img = self._make_image()
        with pytest.raises(ValueError, match="shape"):
            get_cutout(np.ones((2, 2, 2)), img)

    def test_1d_image_raises(self):
        with pytest.raises(ValueError, match="dimension"):
            get_cutout(np.array([[0, 0], [5, 5]]), np.array([1, 2, 3]))

    def test_grayscale_image(self):
        img = np.zeros((50, 60), dtype=np.uint8)
        points = np.array([[5, 10], [20, 30]])
        cutout = get_cutout(points, img)
        assert cutout.shape == (20, 15)

    def test_list_input_converted(self):
        img = self._make_image(h=50, w=50)
        points = [[5, 10], [20, 30]]
        cutout = get_cutout(points, img)
        assert cutout.shape == (20, 15, 3)

    def test_float_coordinates(self):
        img = self._make_image(h=50, w=50)
        points = np.array([[5.7, 10.3], [20.1, 30.9]])
        cutout = get_cutout(points, img)
        # int truncation: x 5..20 = 15, y 10..30 = 20
        assert cutout.shape == (20, 15, 3)

    def test_all_coords_outside_returns_empty(self, caplog):
        img = self._make_image(h=50, w=50)
        points = np.array([[200, 300], [400, 500]])
        with caplog.at_level(logging.WARNING):
            cutout = get_cutout(points, img)
        assert cutout.size == 0


# ---------------------------------------------------------------------------
# get_terminal_size
# ---------------------------------------------------------------------------
class TestGetTerminalSize:
    """Tests for get_terminal_size."""

    def test_returns_tuple_of_two_ints(self):
        result = get_terminal_size()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_default_values_used_on_os_error(self):
        with mock.patch("os.get_terminal_size", side_effect=OSError):
            result = get_terminal_size()
        assert result == (80, 24)

    def test_custom_default(self):
        with mock.patch("os.get_terminal_size", side_effect=OSError):
            result = get_terminal_size(default=(120, 40))
        assert result == (120, 40)

    def test_successful_query_returns_real_size(self):
        fake_size = (132, 50)
        with mock.patch("os.get_terminal_size", return_value=fake_size):
            result = get_terminal_size()
        assert result == fake_size

    def test_first_fd_fails_second_succeeds(self):
        """If fd 0 raises OSError but fd 1 succeeds, use fd 1's result."""
        fake_size = (99, 33)
        call_count = 0

        def side_effect(fd):
            nonlocal call_count
            call_count += 1
            if fd == 0:
                raise OSError("stdin not a terminal")
            return fake_size

        with mock.patch("os.get_terminal_size", side_effect=side_effect):
            result = get_terminal_size()
        assert result == fake_size


# ---------------------------------------------------------------------------
# print_objects_as_table
# ---------------------------------------------------------------------------
class TestPrintObjectsAsTable:
    """Tests for print_objects_as_table attribute-safety and output."""

    def _make_obj(self, **kwargs):
        class _Obj:
            pass

        o = _Obj()
        for k, v in kwargs.items():
            setattr(o, k, v)
        return o

    def test_full_object(self, capsys):
        obj = self._make_obj(
            id=1, age=5, hit_counter=3, last_distance=0.1234, initializing_id=0
        )
        print_objects_as_table([obj])
        out = capsys.readouterr().out
        assert "1" in out

    def test_missing_attributes_shows_fallback(self, capsys):
        obj = self._make_obj()
        print_objects_as_table([obj])
        out = capsys.readouterr().out
        assert "?" in out

    def test_partial_attributes(self, capsys):
        obj = self._make_obj(id=42, age=7)
        print_objects_as_table([obj])
        out = capsys.readouterr().out
        assert "42" in out
        assert "?" in out

    def test_empty_sequence_no_error(self, capsys):
        print_objects_as_table([])
        # Should not raise; just prints an empty table
        out = capsys.readouterr().out
        assert isinstance(out, str)

    def test_multiple_objects(self, capsys):
        objs = [
            self._make_obj(id=1, age=2, hit_counter=1, last_distance=0.5, initializing_id=0),
            self._make_obj(id=2, age=3, hit_counter=2, last_distance=1.0, initializing_id=1),
        ]
        print_objects_as_table(objs)
        out = capsys.readouterr().out
        assert "1" in out
        assert "2" in out

    def test_none_last_distance_shows_question_mark(self, capsys):
        obj = self._make_obj(id=1, age=1, hit_counter=1, last_distance=None, initializing_id=0)
        print_objects_as_table([obj])
        out = capsys.readouterr().out
        assert "?" in out


# ---------------------------------------------------------------------------
# DummyOpenCVImport
# ---------------------------------------------------------------------------
class TestDummyOpenCVImport:
    """Tests for DummyOpenCVImport placeholder."""

    def test_attribute_access_raises_import_error(self):
        dummy = DummyOpenCVImport()
        with pytest.raises(ImportError, match="OpenCV"):
            _ = dummy.VideoCapture

    def test_any_attribute_raises(self):
        dummy = DummyOpenCVImport()
        with pytest.raises(ImportError):
            _ = dummy.imread

    def test_error_mentions_install_command(self):
        dummy = DummyOpenCVImport()
        with pytest.raises(ImportError, match="pip install"):
            _ = dummy.something


# ---------------------------------------------------------------------------
# DummyMOTMetricsImport
# ---------------------------------------------------------------------------
class TestDummyMOTMetricsImport:
    """Tests for DummyMOTMetricsImport placeholder."""

    def test_attribute_access_raises_import_error(self):
        dummy = DummyMOTMetricsImport()
        with pytest.raises(ImportError, match="metrics"):
            _ = dummy.MOTAccumulator

    def test_any_attribute_raises(self):
        dummy = DummyMOTMetricsImport()
        with pytest.raises(ImportError):
            _ = dummy.create

    def test_error_mentions_install_command(self):
        dummy = DummyMOTMetricsImport()
        with pytest.raises(ImportError, match="pip install"):
            _ = dummy.anything


# ---------------------------------------------------------------------------
# warn_once
# ---------------------------------------------------------------------------
class TestWarnOnce:
    """Tests for warn_once caching behavior."""

    def setup_method(self):
        """Clear the warn_once cache before each test."""
        warn_once.cache_clear()

    def test_warns_on_first_call(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_once("test-unique-message-alpha")
        assert "test-unique-message-alpha" in caplog.text

    def test_second_call_same_message_no_extra_warning(self, caplog):
        warn_once.cache_clear()
        with caplog.at_level(logging.WARNING):
            warn_once("test-unique-message-beta")
        first_count = caplog.text.count("test-unique-message-beta")

        with caplog.at_level(logging.WARNING):
            warn_once("test-unique-message-beta")
        second_count = caplog.text.count("test-unique-message-beta")

        assert second_count == first_count  # no new warning emitted

    def test_different_messages_both_warn(self, caplog):
        warn_once.cache_clear()
        with caplog.at_level(logging.WARNING):
            warn_once("msg-gamma-1")
            warn_once("msg-gamma-2")
        assert "msg-gamma-1" in caplog.text
        assert "msg-gamma-2" in caplog.text

    def test_cache_clear_allows_rewarn(self, caplog):
        with caplog.at_level(logging.WARNING):
            warn_once("msg-delta")
        count_after_first = caplog.text.count("msg-delta")

        warn_once.cache_clear()

        with caplog.at_level(logging.WARNING):
            warn_once("msg-delta")
        count_after_second = caplog.text.count("msg-delta")

        assert count_after_second == count_after_first + 1
