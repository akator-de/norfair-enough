"""MOTChallenge-style I/O helpers and accumulators for the evaluation suite."""

import logging
import os
from collections import OrderedDict

import numpy as np
from rich.progress import track

from norfair.tracker import Detection

_logger = logging.getLogger(__name__)

# MotMetrics and pandas are optional dependencies
try:
    import motmetrics as mm
    import pandas as pd
except ImportError:
    from .utils import DummyMOTMetricsImport

    mm = DummyMOTMetricsImport()  # type: ignore[assignment]
    pd = DummyMOTMetricsImport()  # type: ignore[assignment]


class InformationFile:
    """Tiny reader for MOTChallenge ``seqinfo.ini`` style metadata files.

    Loads the file once at construction time and exposes a simple
    :meth:`search` method to look up ``key=value`` pairs.

    Parameters
    ----------
    file_path : str
        Path to the ``seqinfo.ini`` (or similarly formatted) file.

    """

    def __init__(self, file_path: str):
        """Read ``file_path`` into memory and split it into lines."""
        self.path = file_path
        with open(file_path) as myfile:
            file = myfile.read()
        self.lines = file.splitlines()

    def search(self, variable_name: str) -> int | str:
        """Return the value of ``variable_name`` in the loaded file.

        Integer-looking values are returned as ``int``; everything
        else is returned as ``str``.

        Parameters
        ----------
        variable_name : str
            Key to look up, matched as a line prefix followed by an
            ``=``.

        Returns
        -------
        int or str
            The parsed value.

        Raises
        ------
        ValueError
            If ``variable_name`` is not found.

        """
        result: str
        for line in self.lines:
            if line[: len(variable_name)] == variable_name:
                result = line[len(variable_name) + 1 :]
                break
        else:
            raise ValueError(f"Couldn't find '{variable_name}' in {self.path}")
        if result.isdigit():
            return int(result)
        else:
            return result


class PredictionsTextFile:
    """Write tracked objects to a MOTChallenge-format predictions file.

    Each call to :meth:`update` appends one row per tracked object
    for the current frame; the file is closed automatically once the
    number of frames hits the sequence length, or explicitly via
    :meth:`close`.

    Parameters
    ----------
    input_path : str
        Path to the sequence being processed.
    save_path : str, optional
        Directory under which a ``predictions/`` folder is created
        for the output file.
    information_file : InformationFile, optional
        Pre-parsed ``seqinfo.ini`` wrapper. When omitted, one is
        loaded from ``input_path/seqinfo.ini``.

    """

    def __init__(
        self,
        input_path: str,
        save_path: str = ".",
        information_file: InformationFile | None = None,
    ):
        """Create the output file and record the sequence length."""
        file_name = os.path.split(input_path)[1]

        if information_file is None:
            seqinfo_path = os.path.join(input_path, "seqinfo.ini")
            information_file = InformationFile(file_path=seqinfo_path)

        seq_length = information_file.search(variable_name="seqLength")
        # seqLength should always be an int
        self.length: int = (
            int(seq_length) if isinstance(seq_length, str) else seq_length
        )

        predictions_folder = os.path.join(save_path, "predictions")
        if not os.path.exists(predictions_folder):
            os.makedirs(predictions_folder)

        out_file_name = os.path.join(predictions_folder, file_name + ".txt")
        self.text_file = open(out_file_name, "w+")

        self.frame_number = 1

    def update(self, predictions, frame_number=None):
        """Write ``predictions`` for the current frame to the output file.

        The output line format is::

            frame_number, id, bb_left, bb_top, bb_width, bb_height, -1, -1, -1, -1

        Parameters
        ----------
        predictions : iterable of TrackedObject
            Objects tracked for the current frame.
        frame_number : int, optional
            Override for the frame index. When ``None``, the internal
            counter is used.

        """
        if frame_number is None:
            frame_number = self.frame_number
        for obj in predictions:
            frame_str = str(int(frame_number))
            id_str = str(int(obj.id))
            bb_left_str = str(obj.estimate[0, 0])
            bb_top_str = str(obj.estimate[0, 1])  # [0,1]
            bb_width_str = str(obj.estimate[1, 0] - obj.estimate[0, 0])
            bb_height_str = str(obj.estimate[1, 1] - obj.estimate[0, 1])
            row_text_out = (
                frame_str
                + ","
                + id_str
                + ","
                + bb_left_str
                + ","
                + bb_top_str
                + ","
                + bb_width_str
                + ","
                + bb_height_str
                + ",-1,-1,-1,-1"
            )
            self.text_file.write(row_text_out)
            self.text_file.write("\n")

        self.frame_number += 1

        if self.frame_number > self.length:
            self.text_file.close()

    def close(self):
        """Close the underlying file handle."""
        if hasattr(self, "text_file") and self.text_file and not self.text_file.closed:
            self.text_file.close()

    def __enter__(self):
        """Enter the runtime context and return ``self``."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context, ensuring the file handle is closed."""
        self.close()
        return False

    def __del__(self):
        """Ensure the underlying file is closed on garbage collection."""
        self.close()


class DetectionFileParser:
    """Parse MOTChallenge ``det/det.txt`` files into Norfair detections.

    Pre-sorts detections by frame so that iterating the parser yields
    a list of :class:`Detection` for each frame in order.

    Parameters
    ----------
    input_path : str
        Path to the MOTChallenge sequence directory.
    information_file : InformationFile, optional
        Pre-parsed ``seqinfo.ini`` wrapper. When omitted, one is
        loaded from ``input_path/seqinfo.ini``.

    """

    def __init__(
        self, input_path: str, information_file: InformationFile | None = None
    ):
        """Load and pre-sort the detections matrix."""
        self.frame_number = 1

        # Get detecions matrix data with rows corresponding to:
        # frame, id, bb_left, bb_top, bb_right, bb_down, conf, x, y, z
        detections_path = os.path.join(input_path, "det/det.txt")

        self.matrix_detections = np.loadtxt(detections_path, dtype="f", delimiter=",")
        row_order = np.argsort(self.matrix_detections[:, 0])
        self.matrix_detections = self.matrix_detections[row_order]
        # Coordinates refer to box corners
        self.matrix_detections[:, 4] = (
            self.matrix_detections[:, 2] + self.matrix_detections[:, 4]
        )
        self.matrix_detections[:, 5] = (
            self.matrix_detections[:, 3] + self.matrix_detections[:, 5]
        )

        if information_file is None:
            seqinfo_path = os.path.join(input_path, "seqinfo.ini")
            information_file = InformationFile(file_path=seqinfo_path)

        seq_length = information_file.search(variable_name="seqLength")
        # seqLength should always be an int
        self.length: int = (
            int(seq_length) if isinstance(seq_length, str) else seq_length
        )

        self.sorted_by_frame = []
        for frame_number in range(1, self.length + 1):
            self.sorted_by_frame.append(self.get_dets_from_frame(frame_number))

    def get_dets_from_frame(self, frame_number):
        """Return the list of Norfair ``Detection`` for ``frame_number``.

        Parameters
        ----------
        frame_number : int
            1-based frame index to retrieve detections for.

        Returns
        -------
        list of Detection
            Norfair ``Detection`` objects parsed from the raw detection
            matrix for the requested frame. Empty list if no detections
            exist for that frame.
        """
        indexes = np.argwhere(self.matrix_detections[:, 0] == frame_number)
        detections = []
        if len(indexes) > 0:
            actual_det = self.matrix_detections[indexes]
            actual_det = actual_det.reshape(actual_det.shape[0], actual_det.shape[2])
            for det in actual_det:
                points = np.array([[det[2], det[3]], [det[4], det[5]]])
                conf = det[6]
                new_detection = Detection(points, np.array([conf, conf]))
                detections.append(new_detection)
        self.actual_detections = detections
        return detections

    def __iter__(self):
        """Reset the frame counter and return ``self`` as an iterator."""
        self.frame_number = 1
        return self

    def __next__(self):
        """Return the detection list for the next frame in the sequence."""
        if self.frame_number <= self.length:
            self.frame_number += 1
            # Frame_number is always 1 unit bigger than the corresponding index in self.sorted_by_frame, and
            # also we just incremented the frame_number, so now is 2 units bigger than the corresponding index
            return self.sorted_by_frame[self.frame_number - 2]

        raise StopIteration()


class Accumulators:
    """Collect tracker outputs across sequences for MOT metrics evaluation.

    Each sequence is opened with :meth:`create_accumulator`, fed
    frame-by-frame via :meth:`update`, and finally evaluated with
    :meth:`compute_metrics` to produce a dataframe of metrics.
    """

    def __init__(self):
        """Initialize the per-sequence prediction buffers."""
        self.matrixes_predictions = []
        # One entry per registered sequence.
        self.paths: list[str] = []

    def create_accumulator(self, input_path, information_file=None):
        """Start collecting predictions for the sequence at ``input_path``.

        Parameters
        ----------
        input_path : str
            Path to the MOTChallenge sequence directory.
        information_file : InformationFile, optional
            Pre-parsed ``seqinfo.ini`` wrapper. When omitted, one is
            loaded from ``input_path/seqinfo.ini``.

        """
        # Check that motmetrics is installed here, so we don't have to process
        # the whole dataset before failing out if we don't.
        mm.metrics  # noqa: B018 — intentional attribute access to fail early if motmetrics not installed

        file_name = os.path.split(input_path)[1]

        self.frame_number = 1
        # Save the path of this video
        self.paths.append(input_path)
        # Initialize a matrix where we will save our predictions for this video (in the MOTChallenge format)
        self.matrix_predictions = []

        # Initialize progress bar
        if information_file is None:
            seqinfo_path = os.path.join(input_path, "seqinfo.ini")
            information_file = InformationFile(file_path=seqinfo_path)

        seq_length = information_file.search(variable_name="seqLength")
        # seqLength should always be an int
        length_int: int = int(seq_length) if isinstance(seq_length, str) else seq_length

        self.progress_bar_iter = iter(
            track(range(length_int - 1), description=file_name, transient=False)
        )

    def update(self, predictions=None):
        """Append ``predictions`` for the current frame and advance the bar.

        Parameters
        ----------
        predictions : iterable of TrackedObject, optional
            Objects tracked for the current frame. When ``None``, the
            frame is recorded as empty but still advances the counter.

        """
        # Get the tracked boxes from this frame in an array
        if predictions is not None:
            for obj in predictions:
                new_row = [
                    self.frame_number,
                    obj.id,
                    obj.estimate[0, 0],
                    obj.estimate[0, 1],
                    obj.estimate[1, 0] - obj.estimate[0, 0],
                    obj.estimate[1, 1] - obj.estimate[0, 1],
                    -1,
                    -1,
                    -1,
                    -1,
                ]
                self.matrix_predictions.append(new_row)
        self.frame_number += 1
        # Advance in progress bar
        try:
            next(self.progress_bar_iter)
        except StopIteration:
            self.matrixes_predictions.append(self.matrix_predictions)
            return

    def compute_metrics(
        self,
        metrics=None,
        generate_overall=True,
    ):
        """Compute MOTChallenge metrics over all collected sequences.

        Parameters
        ----------
        metrics : list of str, optional
            Subset of ``motmetrics`` metrics to compute. Defaults to
            the full MOTChallenge list.
        generate_overall : bool, optional
            If ``True``, include an ``OVERALL`` row aggregating every
            sequence.

        Returns
        -------
        pandas.DataFrame
            Dataframe of per-sequence metrics (and overall if
            requested). Also stored on ``self.summary_dataframe``.

        """
        if metrics is None:
            metrics = list(mm.metrics.motchallenge_metrics)

        self.summary_text, self.summary_dataframe = eval_motChallenge(
            matrixes_predictions=self.matrixes_predictions,
            paths=self.paths,
            metrics=metrics,
            generate_overall=generate_overall,
        )

        return self.summary_dataframe

    def save_metrics(self, save_path=".", file_name="metrics.txt"):
        """Write the rendered ``summary_text`` to ``save_path/file_name``.

        Parameters
        ----------
        save_path : str, optional
            Directory where the metrics file is written. Created if it
            does not exist. Defaults to ``"."``.
        file_name : str, optional
            Name of the output file. Defaults to ``"metrics.txt"``.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        metrics_path = os.path.join(save_path, file_name)
        with open(metrics_path, "w") as metrics_file:
            metrics_file.write(self.summary_text)

    def print_metrics(self):
        """Print the rendered ``summary_text`` to stdout."""
        print(self.summary_text)


def load_motchallenge(matrix_data, min_confidence=-1):
    """Load MOTChallenge-formatted predictions from a numpy array.

    Adapted from ``motmetrics.io.loadtxt`` but reading from an
    in-memory array instead of a text file.

    Parameters
    ----------
    matrix_data : np.ndarray
        Float array whose rows contain
        ``[frame, id, X, Y, width, height, conf, classId, visibility]``.
    min_confidence : float, optional
        Rows with ``Confidence < min_confidence`` are dropped. Set
        this to ``1`` when loading MOTChallenge ground truth so
        invalid rectangles are ignored during matching.

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns ``['X', 'Y', 'Width', 'Height',
        'Confidence', 'ClassId', 'Visibility']``, indexed by
        ``('FrameId', 'Id')``.

    """
    df = pd.DataFrame(
        data=matrix_data,
        columns=[
            "FrameId",
            "Id",
            "X",
            "Y",
            "Width",
            "Height",
            "Confidence",
            "ClassId",
            "Visibility",
            "unused",
        ],
    )
    df = df.set_index(["FrameId", "Id"])
    # Account for matlab convention.
    df[["X", "Y"]] -= (1, 1)

    # Removed trailing column
    del df["unused"]

    # Remove all rows without sufficient confidence
    return df[df["Confidence"] >= min_confidence]


def compare_dataframes(gts, ts):
    """Build a ``motmetrics`` accumulator per sequence.

    Parameters
    ----------
    gts : dict of str to pandas.DataFrame
        Mapping of sequence name to ground-truth dataframe.
    ts : dict of str to pandas.DataFrame
        Mapping of sequence name to tracker-output dataframe.

    Returns
    -------
    tuple of (list, list)
        ``(accs, names)`` — the accumulators and the corresponding
        sequence names, in matching order.

    """
    accs = []
    names = []
    for k, tsacc in ts.items():
        _logger.info("Comparing %s ...", k)
        if k in gts:
            accs.append(
                mm.utils.compare_to_groundtruth(gts[k], tsacc, "iou", distth=0.5)
            )
            names.append(k)

    return accs, names


def eval_motChallenge(matrixes_predictions, paths, metrics=None, generate_overall=True):
    """Evaluate tracker predictions against MOTChallenge ground truth.

    Parameters
    ----------
    matrixes_predictions : list of np.ndarray
        Per-sequence prediction arrays in the format accepted by
        :func:`load_motchallenge`.
    paths : sequence of str
        Paths to the corresponding MOTChallenge sequence directories;
        ground truth is read from ``<path>/gt/gt.txt``.
    metrics : list of str, optional
        Metric names to compute. Defaults to the full MOTChallenge
        list.
    generate_overall : bool, optional
        If ``True``, include an ``OVERALL`` row aggregating every
        sequence.

    Returns
    -------
    tuple of (str, pandas.DataFrame)
        ``(summary_text, summary_dataframe)`` — the rendered table
        and the raw metrics dataframe.

    """
    # motmetrics' loadtxt accepts "mot15-2D" as a valid format string
    # The type stubs may be overly restrictive
    fmt_string: str = "mot15-2D"
    gt = OrderedDict(
        [
            (
                os.path.split(p)[1],
                mm.io.loadtxt(
                    os.path.join(p, "gt/gt.txt"), fmt=fmt_string, min_confidence=1
                ),
            )
            for p in paths
        ]
    )

    ts = OrderedDict(
        [
            (os.path.split(paths[n])[1], load_motchallenge(matrixes_predictions[n]))
            for n in range(len(paths))
        ]
    )

    mh = mm.metrics.create()

    accs, names = compare_dataframes(gt, ts)

    if metrics is None:
        metrics = list(mm.metrics.motchallenge_metrics)
    mm.lap.default_solver = "scipy"
    _logger.info("Computing metrics...")
    summary_dataframe = mh.compute_many(
        accs, names=names, metrics=metrics, generate_overall=generate_overall
    )
    summary_text = mm.io.render_summary(
        summary_dataframe,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    return summary_text, summary_dataframe
