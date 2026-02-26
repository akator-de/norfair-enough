"""Tests for norfair.metrics module."""
import os
import tempfile

import numpy as np
import pytest

from norfair.metrics import (
    Accumulators,
    DetectionFileParser,
    InformationFile,
    PredictionsTextFile,
)
from norfair.tracker import Detection


class TestInformationFile:
    """Test InformationFile class for reading MOT seqinfo.ini files."""

    def test_read_valid_file(self, tmp_path):
        """Test reading a valid seqinfo.ini file."""
        seqinfo = tmp_path / "seqinfo.ini"
        seqinfo.write_text(
            "[Sequence]\n"
            "name=test-sequence\n"
            "imDir=img1\n"
            "frameRate=30\n"
            "seqLength=100\n"
            "imWidth=1920\n"
            "imHeight=1080\n"
            "imExt=.jpg\n"
        )

        info = InformationFile(str(seqinfo))
        assert info.search("name") == "test-sequence"
        assert info.search("frameRate") == 30
        assert info.search("seqLength") == 100
        assert info.search("imWidth") == 1920
        assert info.search("imHeight") == 1080

    def test_search_nonexistent_variable(self, tmp_path):
        """Test searching for a variable that doesn't exist."""
        seqinfo = tmp_path / "seqinfo.ini"
        seqinfo.write_text("[Sequence]\nname=test\n")

        info = InformationFile(str(seqinfo))
        with pytest.raises(ValueError, match="Couldn't find"):
            info.search("nonexistent")

    def test_search_string_value(self, tmp_path):
        """Test that string values are returned as strings."""
        seqinfo = tmp_path / "seqinfo.ini"
        seqinfo.write_text("imDir=img1\nimExt=.jpg\n")

        info = InformationFile(str(seqinfo))
        assert info.search("imDir") == "img1"
        assert info.search("imExt") == ".jpg"
        assert isinstance(info.search("imDir"), str)

    def test_search_numeric_value(self, tmp_path):
        """Test that numeric values are converted to int."""
        seqinfo = tmp_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=123\nframeRate=30\n")

        info = InformationFile(str(seqinfo))
        assert info.search("seqLength") == 123
        assert isinstance(info.search("seqLength"), int)
        assert info.search("frameRate") == 30


class TestPredictionsTextFile:
    """Test PredictionsTextFile class for writing MOT predictions."""

    def test_create_predictions_file(self, tmp_path):
        """Test creating a predictions file with proper directory structure."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=10\n")

        save_path = tmp_path / "output"
        ptf = PredictionsTextFile(str(input_path), str(save_path))

        # Check that predictions folder was created
        assert (save_path / "predictions").exists()
        assert ptf.length == 10
        assert ptf.frame_number == 1

        ptf.close()

    def test_write_predictions(self, tmp_path):
        """Test writing predictions to file."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=5\n")

        save_path = tmp_path / "output"
        ptf = PredictionsTextFile(str(input_path), str(save_path))

        # Create mock tracked object
        class MockTrackedObject:
            def __init__(self, obj_id, estimate):
                self.id = obj_id
                self.estimate = estimate

        obj = MockTrackedObject(1, np.array([[10.5, 20.5], [30.5, 40.5]]))
        ptf.update([obj])

        ptf.close()

        # Read the file and check content
        output_file = save_path / "predictions" / "input.txt"
        assert output_file.exists()
        content = output_file.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1
        parts = lines[0].split(",")
        assert parts[0] == "1"  # frame_number
        assert parts[1] == "1"  # id
        assert float(parts[2]) == 10.5  # bb_left
        assert float(parts[3]) == 20.5  # bb_top
        assert float(parts[4]) == 20.0  # bb_width (30.5 - 10.5)
        assert float(parts[5]) == 20.0  # bb_height (40.5 - 20.5)

    def test_update_with_custom_frame_number(self, tmp_path):
        """Test updating with a custom frame number."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=10\n")

        save_path = tmp_path / "output"
        ptf = PredictionsTextFile(str(input_path), str(save_path))

        class MockTrackedObject:
            def __init__(self, obj_id, estimate):
                self.id = obj_id
                self.estimate = estimate

        obj = MockTrackedObject(5, np.array([[1, 2], [3, 4]]))
        ptf.update([obj], frame_number=7)

        ptf.close()

        output_file = save_path / "predictions" / "input.txt"
        content = output_file.read_text()
        lines = content.strip().split("\n")
        parts = lines[0].split(",")
        assert parts[0] == "7"  # frame_number should be 7

    def test_auto_close_on_sequence_end(self, tmp_path):
        """Test that file is closed automatically when sequence ends."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=2\n")

        save_path = tmp_path / "output"
        ptf = PredictionsTextFile(str(input_path), str(save_path))

        class MockTrackedObject:
            def __init__(self, obj_id, estimate):
                self.id = obj_id
                self.estimate = estimate

        obj = MockTrackedObject(1, np.array([[1, 2], [3, 4]]))
        ptf.update([obj])
        assert not ptf.text_file.closed
        ptf.update([obj])
        assert ptf.text_file.closed

    def test_close_idempotent(self, tmp_path):
        """Test that close() can be called multiple times safely."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=5\n")

        save_path = tmp_path / "output"
        ptf = PredictionsTextFile(str(input_path), str(save_path))

        ptf.close()
        assert ptf.text_file.closed

        # Should not raise
        ptf.close()
        assert ptf.text_file.closed

    def test_context_manager_closes_file(self, tmp_path):
        """Test that file is closed when used as context manager (via __del__)."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=5\n")

        save_path = tmp_path / "output"
        ptf = PredictionsTextFile(str(input_path), str(save_path))
        file_handle = ptf.text_file

        del ptf
        # File should be closed by __del__
        assert file_handle.closed


class TestDetectionFileParser:
    """Test DetectionFileParser class for reading MOT detections."""

    def test_parse_detections(self, tmp_path):
        """Test parsing a detection file."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        det_dir = input_path / "det"
        det_dir.mkdir()

        # Create a simple detection file
        # Format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
        det_file = det_dir / "det.txt"
        det_file.write_text(
            "1,-1,100,200,50,60,0.9,-1,-1,-1\n"
            "1,-1,300,400,70,80,0.8,-1,-1,-1\n"
            "2,-1,110,210,50,60,0.85,-1,-1,-1\n"
        )

        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=2\n")

        parser = DetectionFileParser(str(input_path))
        assert parser.length == 2

        # Check first frame detections
        dets_frame_1 = parser.sorted_by_frame[0]
        assert len(dets_frame_1) == 2
        assert isinstance(dets_frame_1[0], Detection)

        # Check that coordinates were converted (width/height added to left/top)
        assert dets_frame_1[0].points[0, 0] == 100
        assert dets_frame_1[0].points[0, 1] == 200
        assert dets_frame_1[0].points[1, 0] == 150  # 100 + 50
        assert dets_frame_1[0].points[1, 1] == 260  # 200 + 60

        # Check scores
        assert dets_frame_1[0].scores[0] == 0.9
        assert dets_frame_1[0].scores[1] == 0.9

    def test_iterator(self, tmp_path):
        """Test that DetectionFileParser can be iterated."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        det_dir = input_path / "det"
        det_dir.mkdir()

        det_file = det_dir / "det.txt"
        det_file.write_text(
            "1,-1,10,20,5,6,0.9,-1,-1,-1\n" "2,-1,11,21,5,6,0.8,-1,-1,-1\n"
        )

        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=2\n")

        parser = DetectionFileParser(str(input_path))

        frames = list(parser)
        assert len(frames) == 2
        assert len(frames[0]) == 1  # frame 1 has 1 detection
        assert len(frames[1]) == 1  # frame 2 has 1 detection

    def test_iterator_reusable(self, tmp_path):
        """Test that iterator can be reset."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        det_dir = input_path / "det"
        det_dir.mkdir()

        det_file = det_dir / "det.txt"
        det_file.write_text("1,-1,10,20,5,6,0.9,-1,-1,-1\n")

        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=1\n")

        parser = DetectionFileParser(str(input_path))

        # First iteration
        frames1 = list(parser)
        assert len(frames1) == 1

        # Second iteration should work
        frames2 = list(parser)
        assert len(frames2) == 1

    def test_empty_frame(self, tmp_path):
        """Test handling frames with no detections."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        det_dir = input_path / "det"
        det_dir.mkdir()

        det_file = det_dir / "det.txt"
        det_file.write_text("1,-1,10,20,5,6,0.9,-1,-1,-1\n")

        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=3\n")

        parser = DetectionFileParser(str(input_path))

        frames = list(parser)
        assert len(frames) == 3
        assert len(frames[0]) == 1
        assert len(frames[1]) == 0  # frame 2 has no detections
        assert len(frames[2]) == 0  # frame 3 has no detections


class TestAccumulators:
    """Test Accumulators class for MOT metrics accumulation."""

    def test_create_accumulator(self, tmp_path):
        """Test creating an accumulator for a sequence."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=5\n")

        # Create dummy gt directory
        gt_dir = input_path / "gt"
        gt_dir.mkdir()
        gt_file = gt_dir / "gt.txt"
        gt_file.write_text("1,1,10,20,5,6,1,-1,-1,-1\n")

        acc = Accumulators()
        # This requires motmetrics, which may not be installed
        try:
            acc.create_accumulator(str(input_path))
            assert acc.frame_number == 1
            assert str(input_path) in acc.paths
        except (ImportError, AttributeError):
            # motmetrics not available, skip this test
            pytest.skip("motmetrics not available")

    def test_update_accumulator(self, tmp_path):
        """Test updating accumulator with predictions."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=3\n")

        gt_dir = input_path / "gt"
        gt_dir.mkdir()
        gt_file = gt_dir / "gt.txt"
        gt_file.write_text("1,1,10,20,5,6,1,-1,-1,-1\n")

        acc = Accumulators()
        try:
            acc.create_accumulator(str(input_path))

            class MockTrackedObject:
                def __init__(self, obj_id, estimate):
                    self.id = obj_id
                    self.estimate = estimate

            obj = MockTrackedObject(1, np.array([[10, 20], [15, 26]]))
            acc.update([obj])
            assert acc.frame_number == 2

            # Update without predictions (empty frame)
            acc.update(None)
            assert acc.frame_number == 3

            # Final update should complete the sequence
            acc.update(None)
            assert len(acc.matrixes_predictions) == 1
        except (ImportError, AttributeError):
            pytest.skip("motmetrics not available")

    def test_accumulator_appends_predictions(self, tmp_path):
        """Test that predictions are accumulated correctly."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=2\n")

        gt_dir = input_path / "gt"
        gt_dir.mkdir()
        gt_file = gt_dir / "gt.txt"
        gt_file.write_text("1,1,10,20,5,6,1,-1,-1,-1\n")

        acc = Accumulators()
        try:
            acc.create_accumulator(str(input_path))

            class MockTrackedObject:
                def __init__(self, obj_id, estimate):
                    self.id = obj_id
                    self.estimate = estimate

            obj1 = MockTrackedObject(1, np.array([[10, 20], [15, 26]]))
            obj2 = MockTrackedObject(2, np.array([[30, 40], [35, 46]]))

            acc.update([obj1, obj2])
            assert len(acc.matrix_predictions) == 2

            acc.update([obj1])
            assert len(acc.matrixes_predictions[0]) == 3  # 2 from frame 1, 1 from frame 2
        except (ImportError, AttributeError):
            pytest.skip("motmetrics not available")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_predictions_text_file_with_no_information_file(self, tmp_path):
        """Test PredictionsTextFile when information_file is passed explicitly."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=5\n")

        info = InformationFile(str(seqinfo))
        save_path = tmp_path / "output"

        ptf = PredictionsTextFile(str(input_path), str(save_path), info)
        assert ptf.length == 5
        ptf.close()

    def test_detection_parser_with_no_information_file(self, tmp_path):
        """Test DetectionFileParser when information_file is passed explicitly."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        det_dir = input_path / "det"
        det_dir.mkdir()
        det_file = det_dir / "det.txt"
        det_file.write_text("1,-1,10,20,5,6,0.9,-1,-1,-1\n")

        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=1\n")
        info = InformationFile(str(seqinfo))

        parser = DetectionFileParser(str(input_path), info)
        assert parser.length == 1

    def test_information_file_string_seqlength(self, tmp_path):
        """Test handling when seqLength is a string that needs conversion."""
        seqinfo = tmp_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=100\n")

        info = InformationFile(str(seqinfo))
        result = info.search("seqLength")
        assert isinstance(result, int)
        assert result == 100

    def test_predictions_with_multiple_objects(self, tmp_path):
        """Test writing predictions for multiple objects in same frame."""
        input_path = tmp_path / "input"
        input_path.mkdir()
        seqinfo = input_path / "seqinfo.ini"
        seqinfo.write_text("seqLength=5\n")

        save_path = tmp_path / "output"
        ptf = PredictionsTextFile(str(input_path), str(save_path))

        class MockTrackedObject:
            def __init__(self, obj_id, estimate):
                self.id = obj_id
                self.estimate = estimate

        objects = [
            MockTrackedObject(1, np.array([[10, 20], [15, 25]])),
            MockTrackedObject(2, np.array([[30, 40], [35, 45]])),
            MockTrackedObject(3, np.array([[50, 60], [55, 65]])),
        ]
        ptf.update(objects)
        ptf.close()

        output_file = save_path / "predictions" / "input.txt"
        content = output_file.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3  # Three objects in one frame