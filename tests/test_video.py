"""Tests for norfair.video module."""
import os
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest


class TestVideoInitialization:
    """Test Video class initialization and validation."""

    @patch("norfair.video.cv2")
    def test_requires_camera_or_input_path(self, mock_cv2):
        """Test that either camera or input_path must be specified."""
        from norfair.video import Video

        with pytest.raises(ValueError, match="must set either"):
            Video()

    @patch("norfair.video.cv2")
    def test_camera_and_input_path_mutually_exclusive(self, mock_cv2):
        """Test that camera and input_path cannot both be specified."""
        from norfair.video import Video

        with pytest.raises(ValueError, match="must set either"):
            Video(camera=0, input_path="video.mp4")

    @patch("norfair.video.cv2")
    def test_camera_must_be_int(self, mock_cv2):
        """Test that camera parameter must be an integer."""
        from norfair.video import Video

        with pytest.raises(ValueError, match="must be an int"):
            Video(camera="0")

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=False)
    def test_input_path_must_exist(self, mock_isfile, mock_cv2):
        """Test that input_path must point to an existing file."""
        from norfair.video import Video

        with pytest.raises(RuntimeError, match="does not exist"):
            Video(input_path="nonexistent.mp4")

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    def test_input_path_must_be_valid_video(self, mock_isfile, mock_cv2):
        """Test that input_path must be a valid video file."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.return_value = 0  # 0 frames = invalid
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_COUNT = 7

        with pytest.raises(RuntimeError, match="does not seem to be a video file"):
            Video(input_path="invalid.mp4")

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test_video.mp4")
    def test_successful_initialization_with_file(
        self, mock_basename, mock_isfile, mock_cv2
    ):
        """Test successful initialization with a video file."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 100,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="test_video.mp4")
        assert video.input_path == "test_video.mp4"
        assert video.output_fps == 30.0
        assert video.input_height == 1080
        assert video.input_width == 1920
        video.close()

    @patch("norfair.video.cv2")
    def test_successful_initialization_with_camera(self, mock_cv2):
        """Test successful initialization with a camera."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FPS: 25.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 720,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1280,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(camera=0)
        assert video.camera == 0
        assert video.output_fps == 25.0
        video.close()

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test.mp4")
    def test_custom_output_fps(self, mock_basename, mock_isfile, mock_cv2):
        """Test setting a custom output FPS."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 100,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="test.mp4", output_fps=60.0)
        assert video.output_fps == 60.0
        video.close()


class TestVideoIteration:
    """Test Video iteration and frame processing."""

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test.mp4")
    def test_iterate_frames(self, mock_basename, mock_isfile, mock_cv2):
        """Test iterating through video frames."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 3,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)

        # Simulate 3 frames then end
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8)
        frame3 = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_capture.read.side_effect = [
            (True, frame1),
            (True, frame2),
            (True, frame3),
            (False, None),
        ]

        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="test.mp4")
        frames = []
        for frame in video:
            frames.append(frame)

        assert len(frames) == 3
        assert np.array_equal(frames[0], frame1)
        assert np.array_equal(frames[1], frame2)
        assert np.array_equal(frames[2], frame3)
        # close() should be called automatically by iterator
        assert video._closed

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test.mp4")
    def test_frame_counter_increments(self, mock_basename, mock_isfile, mock_cv2):
        """Test that frame counter increments correctly."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 2,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_capture.read.side_effect = [(True, frame), (True, frame), (False, None)]

        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="test.mp4")
        assert video.frame_counter == 0

        for idx, _ in enumerate(video, start=1):
            # frame_counter increments during iteration
            pass

        assert video.frame_counter == 2


class TestVideoWrite:
    """Test Video write functionality."""

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test.mp4")
    @patch("os.path.isdir", return_value=True)
    def test_write_creates_video_writer(
        self, mock_isdir, mock_basename, mock_isfile, mock_cv2
    ):
        """Test that write() creates a VideoWriter on first call."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 1,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture

        mock_writer = Mock()
        mock_cv2.VideoWriter.return_value = mock_writer
        mock_cv2.VideoWriter_fourcc.return_value = 123
        mock_cv2.waitKey.return_value = -1

        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="test.mp4")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        video.write(frame)

        mock_cv2.VideoWriter.assert_called_once()
        mock_writer.write.assert_called_once_with(frame)
        video.close()

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test.mp4")
    @patch("os.path.isdir", return_value=True)
    def test_write_multiple_frames(
        self, mock_isdir, mock_basename, mock_isfile, mock_cv2
    ):
        """Test writing multiple frames."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 1,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture

        mock_writer = Mock()
        mock_cv2.VideoWriter.return_value = mock_writer
        mock_cv2.VideoWriter_fourcc.return_value = 123
        mock_cv2.waitKey.return_value = -1

        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="test.mp4")
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8)

        video.write(frame1)
        video.write(frame2)

        assert mock_writer.write.call_count == 2
        video.close()


class TestVideoOutputPath:
    """Test Video output path generation."""

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.splitext")
    @patch("os.path.basename", return_value="input_video.mp4")
    @patch("os.path.isdir", return_value=True)
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    def test_output_path_from_input_file(
        self, mock_join, mock_isdir, mock_basename, mock_splitext, mock_isfile, mock_cv2
    ):
        """Test output path generation from input file."""
        from norfair.video import Video

        mock_splitext.return_value = ("input_video", ".mp4")

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 1,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="input_video.mp4", output_path="/tmp")
        output_path = video.get_output_file_path()

        assert "input_video_out.mp4" in output_path
        video.close()

    @patch("norfair.video.cv2")
    def test_output_path_from_camera(self, mock_cv2):
        """Test output path generation from camera."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        with patch("os.path.isdir", return_value=True):
            with patch("os.path.join", side_effect=lambda *args: "/".join(args)):
                video = Video(camera=0, output_path="/tmp")
                output_path = video.get_output_file_path()

                assert "camera_0" in output_path
                video.close()

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test.mp4")
    @patch("os.path.isdir", return_value=False)
    def test_output_path_as_file(self, mock_isdir, mock_basename, mock_isfile, mock_cv2):
        """Test using output_path as a direct file path."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 1,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="test.mp4", output_path="/tmp/output.avi")
        output_path = video.get_output_file_path()

        assert output_path == "/tmp/output.avi"
        video.close()


class TestVideoCodec:
    """Test Video codec selection."""

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test.mp4")
    @patch("os.path.splitext")
    def test_default_codec_mp4(
        self, mock_splitext, mock_basename, mock_isfile, mock_cv2
    ):
        """Test default codec for .mp4 files."""
        from norfair.video import Video

        mock_splitext.return_value = ("output", ".mp4")

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 1,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="test.mp4")
        codec = video.get_codec_fourcc("output.mp4")

        assert codec == "mp4v"
        video.close()

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test.mp4")
    @patch("os.path.splitext")
    def test_default_codec_avi(
        self, mock_splitext, mock_basename, mock_isfile, mock_cv2
    ):
        """Test default codec for .avi files."""
        from norfair.video import Video

        mock_splitext.return_value = ("output", ".avi")

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 1,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="test.mp4")
        codec = video.get_codec_fourcc("output.avi")

        assert codec == "XVID"
        video.close()

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test.mp4")
    def test_custom_codec(self, mock_basename, mock_isfile, mock_cv2):
        """Test setting a custom codec."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 1,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="test.mp4", output_fourcc="H264")
        codec = video.get_codec_fourcc("any_file.mp4")

        assert codec == "H264"
        video.close()

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test.mp4")
    @patch("os.path.splitext")
    def test_unsupported_extension_raises(
        self, mock_splitext, mock_basename, mock_isfile, mock_cv2
    ):
        """Test that unsupported extensions raise an error."""
        from norfair.video import Video

        mock_splitext.return_value = ("output", ".mkv")

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 1,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="test.mp4")

        with pytest.raises(RuntimeError, match="Could not determine video codec"):
            video.get_codec_fourcc("output.mkv")

        video.close()


class TestVideoContextManager:
    """Test Video context manager support."""

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test.mp4")
    def test_context_manager_calls_close(
        self, mock_basename, mock_isfile, mock_cv2
    ):
        """Test that context manager calls close on exit."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 1,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        with Video(input_path="test.mp4") as video:
            assert not video._closed

        assert video._closed
        mock_capture.release.assert_called_once()

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test.mp4")
    def test_close_idempotent(self, mock_basename, mock_isfile, mock_cv2):
        """Test that close() can be called multiple times safely."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 1,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="test.mp4")
        video.close()
        video.close()  # Should not raise

        assert video._closed
        mock_capture.release.assert_called_once()


class TestVideoShow:
    """Test Video show functionality."""

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test.mp4")
    def test_show_displays_frame(self, mock_basename, mock_isfile, mock_cv2):
        """Test that show() displays a frame."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 1,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.imshow = Mock()
        mock_cv2.waitKey.return_value = -1
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="test.mp4")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        video.show(frame)

        mock_cv2.imshow.assert_called_once()
        mock_cv2.waitKey.assert_called_once_with(1)
        video.close()

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test.mp4")
    def test_show_with_downsample(self, mock_basename, mock_isfile, mock_cv2):
        """Test that show() downsamples when requested."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 1,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.imshow = Mock()
        mock_cv2.resize = Mock(return_value=np.zeros((50, 50, 3), dtype=np.uint8))
        mock_cv2.waitKey.return_value = -1
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="test.mp4")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        video.show(frame, downsample_ratio=2.0)

        mock_cv2.resize.assert_called_once()
        video.close()


class TestVideoEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.expanduser")
    @patch("os.path.basename", return_value="test.mp4")
    def test_tilde_expansion_in_path(
        self, mock_basename, mock_expanduser, mock_isfile, mock_cv2
    ):
        """Test that ~ in paths is expanded."""
        from norfair.video import Video

        mock_expanduser.return_value = "/home/user/test.mp4"

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 1,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="~/test.mp4")
        assert video.input_path == "/home/user/test.mp4"
        video.close()

    @patch("norfair.video.cv2")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.basename", return_value="test.mp4")
    def test_custom_output_extension(self, mock_basename, mock_isfile, mock_cv2):
        """Test setting a custom output extension."""
        from norfair.video import Video

        mock_capture = Mock()
        mock_capture.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 1,
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 100,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_capture
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3

        video = Video(input_path="test.mp4", output_extension="avi")
        assert video.output_extension == "avi"
        video.close()