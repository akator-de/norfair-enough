import os
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest


# Create a mock cv2 module and inject it before importing Video
@pytest.fixture(scope="function", autouse=True)
def mock_cv2():
    """Mock OpenCV for testing without actual video files."""
    # Create mock cv2
    mock = Mock()

    # Mock VideoCapture
    mock_capture = Mock()
    mock_capture.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
    mock_capture.get.side_effect = lambda prop: {
        0: 10,  # CAP_PROP_FRAME_COUNT
        5: 30.0,  # CAP_PROP_FPS
        3: 100.0,  # CAP_PROP_FRAME_WIDTH
        4: 100.0,  # CAP_PROP_FRAME_HEIGHT
    }.get(prop, 0)
    mock_capture.release = Mock()
    mock.VideoCapture.return_value = mock_capture
    mock.CAP_PROP_FRAME_COUNT = 0
    mock.CAP_PROP_FPS = 5
    mock.CAP_PROP_FRAME_WIDTH = 3
    mock.CAP_PROP_FRAME_HEIGHT = 4

    # Mock VideoWriter
    mock_writer = Mock()
    mock_writer.write = Mock()
    mock_writer.release = Mock()
    mock.VideoWriter.return_value = mock_writer
    mock.VideoWriter_fourcc.return_value = 0x00000000

    # Mock other functions
    mock.waitKey.return_value = -1
    mock.imshow = Mock()
    mock.resize = Mock(return_value=np.zeros((50, 50, 3), dtype=np.uint8))
    mock.destroyAllWindows = Mock()

    # Inject into sys.modules
    old_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = mock

    # Reload norfair.video to pick up the mocked cv2
    if "norfair.video" in sys.modules:
        import importlib

        import norfair.video

        importlib.reload(norfair.video)

    yield mock

    # Restore
    if old_cv2 is not None:
        sys.modules["cv2"] = old_cv2
    elif "cv2" in sys.modules:
        del sys.modules["cv2"]

    # Reload again to restore original state
    if "norfair.video" in sys.modules:
        import importlib

        import norfair.video

        importlib.reload(norfair.video)


def test_video_requires_input_source():
    """Test that Video requires either camera or input_path."""
    from norfair.video import Video

    with pytest.raises(
        ValueError, match="You must set either 'camera' or 'input_path'"
    ):
        Video()

    with pytest.raises(
        ValueError, match="You must set either 'camera' or 'input_path'"
    ):
        Video(camera=0, input_path="video.mp4")


def test_video_camera_must_be_int():
    """Test that camera parameter must be an integer."""
    from norfair.video import Video

    with pytest.raises(ValueError, match="must be an int"):
        Video(camera="0")


def test_video_context_manager(mock_cv2, tmp_path):
    """Test that Video works as a context manager."""
    from norfair.video import Video

    # Create a fake input file
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        with Video(input_path=str(input_file)) as video:
            assert video is not None
            assert not video._closed

        # After exiting context, should be closed
        assert video._closed


def test_video_close_releases_resources(mock_cv2, tmp_path):
    """Test that close() releases video resources."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file))
        video.close()

        # Verify resources were released
        assert video._closed
        video.video_capture.release.assert_called_once()
        mock_cv2.destroyAllWindows.assert_called_once()


def test_video_close_is_idempotent(mock_cv2, tmp_path):
    """Test that calling close() multiple times is safe."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file))
        video.close()
        video.close()  # Should not raise

        # Should only release once
        assert video.video_capture.release.call_count == 1


def test_video_iteration_calls_close(mock_cv2, tmp_path):
    """Test that iterating through video calls close() at the end."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file))

        # Simulate reading all frames
        mock_cv2.VideoCapture.return_value.read.side_effect = [
            (True, np.zeros((100, 100, 3))),
            (False, None),
        ]

        frames = list(video)
        assert len(frames) == 1
        assert video._closed


def test_video_context_manager_early_break(mock_cv2, tmp_path):
    """Test that breaking early from video iteration still releases resources."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        with Video(input_path=str(input_file)) as video:
            # Break after first frame
            for _frame in video:
                break

        # Resources should still be released
        assert video._closed


def test_video_get_output_file_path_with_directory(mock_cv2, tmp_path):
    """Test get_output_file_path when output_path is a directory."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file), output_path=str(output_dir))
        output_path = video.get_output_file_path()

        assert output_path == str(output_dir / "input_out.mp4")


def test_video_get_output_file_path_with_file(mock_cv2, tmp_path):
    """Test get_output_file_path when output_path is a file."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()
    output_file = tmp_path / "custom_output.mp4"

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file), output_path=str(output_file))
        output_path = video.get_output_file_path()

        assert output_path == str(output_file)


def test_video_get_output_file_path_with_camera(mock_cv2, tmp_path):
    """Test get_output_file_path when using camera input."""
    from norfair.video import Video

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    video = Video(camera=0, output_path=str(output_dir))
    output_path = video.get_output_file_path()

    assert output_path == str(output_dir / "camera_0_out.mp4")


def test_video_get_codec_fourcc_mp4(mock_cv2, tmp_path):
    """Test get_codec_fourcc returns correct codec for mp4."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file))
        fourcc = video.get_codec_fourcc("output.mp4")

        assert fourcc == "mp4v"


def test_video_get_codec_fourcc_avi(mock_cv2, tmp_path):
    """Test get_codec_fourcc returns correct codec for avi."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file))
        fourcc = video.get_codec_fourcc("output.avi")

        assert fourcc == "XVID"


def test_video_get_codec_fourcc_custom(mock_cv2, tmp_path):
    """Test get_codec_fourcc with custom fourcc."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file), output_fourcc="avc1")
        fourcc = video.get_codec_fourcc("output.mp4")

        assert fourcc == "avc1"


def test_video_get_codec_fourcc_unsupported_extension(mock_cv2, tmp_path):
    """Test get_codec_fourcc raises error for unsupported extension."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file))
        with pytest.raises(RuntimeError, match="Could not determine video codec"):
            video.get_codec_fourcc("output.xyz")


def test_video_write_creates_video_writer(mock_cv2, tmp_path):
    """Test that write() creates VideoWriter on first call."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file), output_path=str(output_dir))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        assert video.output_video is None
        video.write(frame)
        assert video.output_video is not None

        # Verify VideoWriter was called correctly
        mock_cv2.VideoWriter.assert_called_once()


def test_video_write_calls_video_writer(mock_cv2, tmp_path):
    """Test that write() calls VideoWriter.write()."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file), output_path=str(output_dir))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        video.write(frame)
        mock_cv2.VideoWriter.return_value.write.assert_called_once()


def test_video_show_calls_imshow(mock_cv2, tmp_path):
    """Test that show() calls cv2.imshow()."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        video.show(frame)
        mock_cv2.imshow.assert_called_once()


def test_video_show_with_downsample(mock_cv2, tmp_path):
    """Test that show() with downsample_ratio calls cv2.resize()."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        video.show(frame, downsample_ratio=2.0)
        mock_cv2.resize.assert_called_once()


def test_video_show_without_downsample(mock_cv2, tmp_path):
    """Test that show() without downsample doesn't call cv2.resize()."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        video.show(frame, downsample_ratio=1.0)
        mock_cv2.resize.assert_not_called()


def test_video_expanduser_for_tilde_path(mock_cv2):
    """Test that ~ in input_path is expanded."""
    from norfair.video import Video

    with patch("os.path.expanduser") as mock_expanduser:
        mock_expanduser.return_value = "/home/user/video.mp4"
        with patch("os.path.isfile", return_value=True):
            Video(input_path="~/video.mp4")
            mock_expanduser.assert_called_once_with("~/video.mp4")


def test_video_nonexistent_file_raises_error(mock_cv2):
    """Test that nonexistent input file raises error."""
    from norfair.video import Video

    with (
        patch("os.path.isfile", return_value=False),
        pytest.raises(RuntimeError, match="does not exist"),
    ):
        Video(input_path="/nonexistent/video.mp4")


def test_video_invalid_file_raises_error(mock_cv2, tmp_path):
    """Test that invalid video file raises error."""
    from norfair.video import Video

    input_file = tmp_path / "invalid.mp4"
    input_file.touch()

    # Mock VideoCapture to return 0 frames (invalid file)
    with patch("os.path.isfile", return_value=True):
        mock_cv2.VideoCapture.return_value.get.side_effect = lambda prop: 0

        with pytest.raises(RuntimeError, match="does not seem to be a video file"):
            Video(input_path=str(input_file))


def test_video_output_fps_from_parameter(mock_cv2, tmp_path):
    """Test that output_fps parameter overrides input fps."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file), output_fps=60.0)
        assert video.output_fps == 60.0


def test_video_output_fps_from_input(mock_cv2, tmp_path):
    """Test that output_fps defaults to input fps."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file))
        assert video.output_fps == 30.0  # From mock


def test_video_camera_input(mock_cv2):
    """Test that camera input works."""
    from norfair.video import Video

    video = Video(camera=0)
    assert video.camera == 0
    assert video.input_path is None
    mock_cv2.VideoCapture.assert_called_with(0)


def test_video_frame_counter_increments(mock_cv2, tmp_path):
    """Test that frame_counter increments during iteration."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file))

        # Simulate reading 3 frames
        mock_cv2.VideoCapture.return_value.read.side_effect = [
            (True, np.zeros((100, 100, 3))),
            (True, np.zeros((100, 100, 3))),
            (True, np.zeros((100, 100, 3))),
            (False, None),
        ]

        assert video.frame_counter == 0
        for i, _frame in enumerate(video, start=1):
            assert video.frame_counter == i


def test_video_output_extension_parameter(mock_cv2, tmp_path):
    """Test that output_extension parameter is used."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("os.path.isfile", return_value=True):
        video = Video(
            input_path=str(input_file),
            output_path=str(output_dir),
            output_extension="avi",
        )
        output_path = video.get_output_file_path()

        assert output_path.endswith(".avi")


def test_video_write_releases_on_close(mock_cv2, tmp_path):
    """Test that output video is released when Video is closed."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file), output_path=str(output_dir))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        video.write(frame)

        mock_writer = mock_cv2.VideoWriter.return_value
        video.close()

        # Verify output video was released
        mock_writer.release.assert_called_once()


def test_video_label_in_progress_bar(mock_cv2, tmp_path):
    """Test that label appears in progress bar description."""
    from norfair.video import Video

    input_file = tmp_path / "input.mp4"
    input_file.touch()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file), label="Test Label")
        assert video.label == "Test Label"


def test_video_get_output_file_path_uses_basename(mock_cv2, tmp_path):
    """Test that output filename uses basename of input path."""
    from norfair.video import Video

    input_file = tmp_path / "subdir" / "video.mp4"
    input_file.parent.mkdir(parents=True)
    input_file.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("os.path.isfile", return_value=True):
        video = Video(input_path=str(input_file), output_path=str(output_dir))
        output_path = video.get_output_file_path()

        # Should use only the basename, not the full path
        assert os.path.basename(output_path) == "video_out.mp4"
