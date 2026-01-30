import numpy as np
import pytest

from norfair.drawing.color import (
    Color,
    Palette,
    hex_to_bgr,
    parse_color,
)
from norfair.drawing.drawer import Drawable
from norfair.drawing.utils import _build_text, _centroid
from norfair.tracker import Detection

# ---------------------------------------------------------------------------
# hex_to_bgr
# ---------------------------------------------------------------------------


class TestHexToBgr:
    def test_6_digit(self):
        assert hex_to_bgr("#010203") == (3, 2, 1)
        assert hex_to_bgr("#ffffff") == (255, 255, 255)
        assert hex_to_bgr("#000000") == (0, 0, 0)

    def test_3_digit(self):
        # #123 → #112233 → BGR (0x33, 0x22, 0x11) = (51, 34, 17)
        assert hex_to_bgr("#123") == (51, 34, 17)
        assert hex_to_bgr("#fff") == (255, 255, 255)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            hex_to_bgr("not a color")
        with pytest.raises(ValueError):
            hex_to_bgr("#gggggg")
        with pytest.raises(ValueError):
            hex_to_bgr("#12345")  # 5 digits


# ---------------------------------------------------------------------------
# parse_color
# ---------------------------------------------------------------------------


class TestParseColor:
    def test_hex_string(self):
        assert parse_color("#ff0000") == (0, 0, 255)  # red in BGR

    def test_named_color(self):
        assert parse_color("red") == Color.red
        assert parse_color("blue") == Color.blue

    def test_tuple_passthrough(self):
        assert parse_color((10, 20, 30)) == (10, 20, 30)

    def test_invalid_name_raises(self):
        with pytest.raises(AttributeError):
            parse_color("not_a_color_name")


# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------


class TestColor:
    def test_well_known_colors(self):
        assert Color.black == (0, 0, 0)
        assert Color.white == (255, 255, 255)
        assert Color.red == (0, 0, 255)  # BGR
        assert Color.green == (0, 128, 0)
        assert Color.blue == (255, 0, 0)  # BGR

    def test_tab_colors_exist(self):
        for i in range(1, 21):
            color = getattr(Color, f"tab{i}")
            assert isinstance(color, tuple)
            assert len(color) == 3

    def test_colorblind_colors_exist(self):
        for i in range(1, 11):
            color = getattr(Color, f"cb{i}")
            assert isinstance(color, tuple)
            assert len(color) == 3


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------


class TestPalette:
    def setup_method(self):
        """Reset palette to default before each test."""
        Palette.set("tab10")
        Palette.set_default_color(Color.black)

    def test_set_named_palette(self):
        Palette.set("tab20")
        # tab20 has 20 colors, different ids should get different colors
        colors = {Palette.choose_color(i) for i in range(20)}
        assert len(colors) == 20

    def test_set_colorblind_palette(self):
        Palette.set("colorblind")
        colors = {Palette.choose_color(i) for i in range(10)}
        assert len(colors) == 10

    def test_set_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Invalid palette name"):
            Palette.set("nonexistent")

    def test_set_custom_palette(self):
        custom = [Color.red, Color.blue, "#00ff00"]
        Palette.set(custom)
        # With 3 colors, hash(0) % 3 picks one
        result = Palette.choose_color(0)
        assert result in [Color.red, Color.blue, parse_color("#00ff00")]

    def test_choose_color_none_returns_default(self):
        assert Palette.choose_color(None) == Color.black

    def test_set_default_color(self):
        Palette.set_default_color(Color.red)
        assert Palette.choose_color(None) == Color.red

    def test_choose_color_deterministic(self):
        c1 = Palette.choose_color(42)
        c2 = Palette.choose_color(42)
        assert c1 == c2

    def test_choose_color_with_string(self):
        # Strings are hashable, should work
        c = Palette.choose_color("person")
        assert isinstance(c, tuple) and len(c) == 3


# ---------------------------------------------------------------------------
# _centroid
# ---------------------------------------------------------------------------


class TestCentroid:
    def test_single_point(self):
        points = np.array([[100, 200]])
        assert _centroid(points) == (100, 200)

    def test_multiple_points(self):
        points = np.array([[0, 0], [10, 20], [20, 40]])
        assert _centroid(points) == (10, 20)

    def test_rounding(self):
        points = np.array([[0, 0], [1, 1]])
        # mean is (0.5, 0.5), int truncates to (0, 0)
        assert _centroid(points) == (0, 0)


# ---------------------------------------------------------------------------
# _build_text
# ---------------------------------------------------------------------------


class TestBuildText:
    def _make_drawable(self, label=None, id=None, scores=None):
        """Create a minimal Drawable for testing _build_text."""
        points = np.array([[0, 0]])
        return Drawable(
            points=points,
            label=label,
            id=id,
            scores=scores,
            live_points=np.array([True]),
        )

    def test_empty(self):
        d = self._make_drawable()
        assert (
            _build_text(d, draw_labels=False, draw_ids=False, draw_scores=False) == ""
        )

    def test_label_only(self):
        d = self._make_drawable(label="car")
        assert (
            _build_text(d, draw_labels=True, draw_ids=False, draw_scores=False) == "car"
        )

    def test_id_only(self):
        d = self._make_drawable(id=7)
        assert (
            _build_text(d, draw_labels=False, draw_ids=True, draw_scores=False) == "7"
        )

    def test_scores_only(self):
        d = self._make_drawable(scores=np.array([0.95]))
        text = _build_text(d, draw_labels=False, draw_ids=False, draw_scores=True)
        assert text == "0.95"

    def test_label_and_id(self):
        d = self._make_drawable(label="person", id=3)
        text = _build_text(d, draw_labels=True, draw_ids=True, draw_scores=False)
        assert text == "person-3"

    def test_all(self):
        d = self._make_drawable(label="car", id=5, scores=np.array([0.88]))
        text = _build_text(d, draw_labels=True, draw_ids=True, draw_scores=True)
        assert text == "car-5-0.88"

    def test_none_fields_ignored(self):
        d = self._make_drawable(label=None, id=None, scores=None)
        text = _build_text(d, draw_labels=True, draw_ids=True, draw_scores=True)
        assert text == ""


# ---------------------------------------------------------------------------
# Drawable
# ---------------------------------------------------------------------------


class TestDrawable:
    def test_from_detection(self):
        det = Detection(
            points=np.array([[10, 20], [30, 40]]),
            scores=np.array([0.9, 0.8]),
            label="person",
        )
        d = Drawable(det)
        np.testing.assert_array_equal(d.points, det.points)
        assert d.id is None
        assert d.label == "person"
        np.testing.assert_array_equal(d.scores, np.array([0.9, 0.8]))
        assert d.live_points.all()  # all alive for detections

    def test_from_none_with_kwargs(self):
        pts = np.array([[1, 2]])
        lp = np.array([True])
        d = Drawable(points=pts, id=42, label="car", live_points=lp)
        np.testing.assert_array_equal(d.points, pts)
        assert d.id == 42
        assert d.label == "car"
        np.testing.assert_array_equal(d.live_points, lp)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Extecting"):
            Drawable("not a valid object")

    def test_detection_live_points_shape(self):
        det = Detection(points=np.array([[0, 0], [1, 1], [2, 2]]))
        d = Drawable(det)
        assert d.live_points.shape == (3,)
        assert d.live_points.dtype == bool
