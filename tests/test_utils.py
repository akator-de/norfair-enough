"""Tests for norfair.utils -- get_cutout and print_objects_as_table."""

import logging

import numpy as np
import pytest

from norfair.utils import get_cutout, print_objects_as_table


class TestGetCutout:
    """Tests for get_cutout input validation and clipping."""

    def _make_image(self, h=100, w=200):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_normal_cutout(self):
        img = self._make_image()
        points = np.array([[10, 20], [50, 60]])
        cutout = get_cutout(points, img)
        assert cutout.shape == (40, 40, 3)

    def test_empty_points_raises(self):
        img = self._make_image()
        with pytest.raises(ValueError, match="empty"):
            get_cutout(np.empty((0, 2)), img)

    def test_wrong_ndim_1d_raises(self):
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

    def test_coords_clipped_to_image_bounds(self):
        img = self._make_image(h=50, w=50)
        points = np.array([[-10, -10], [100, 100]])
        cutout = get_cutout(points, img)
        assert cutout.shape == (50, 50, 3)

    def test_degenerate_same_point_warns(self, caplog):
        img = self._make_image()
        points = np.array([[10, 20], [10, 20]])
        with caplog.at_level(logging.WARNING):
            cutout = get_cutout(points, img)
        assert cutout.size == 0
        assert "degenerate" in caplog.text


class TestPrintObjectsAsTable:
    """Tests for print_objects_as_table attribute-safety."""

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

    def test_empty_sequence(self, capsys):
        print_objects_as_table([])
