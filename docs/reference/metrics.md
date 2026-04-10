# Metrics

Utilities for evaluating a Norfair pipeline against the
[MOTChallenge](https://motchallenge.net/) format. These let you parse
ground-truth annotations, record per-frame predictions, and compute standard
multi-object-tracking metrics (MOTA, IDF1, ID switches, …) via
[`py-motmetrics`](https://github.com/cheind/py-motmetrics).

This module is optional. Its MOTChallenge-specific helpers require the
`metrics` extra:

```bash
pip install 'norfair-enough[metrics]'
```

## Overview

- [`DetectionFileParser`][norfair.metrics.DetectionFileParser] — reads
  MOTChallenge `det.txt` / `gt.txt` files and yields per-frame `Detection`
  lists ready to feed into a [`Tracker`][norfair.tracker.Tracker].
- [`PredictionsTextFile`][norfair.metrics.PredictionsTextFile] — writes your
  tracker output to a MOTChallenge-format text file, one row per tracked
  object per frame.
- [`InformationFile`][norfair.metrics.InformationFile] — tiny parser for the
  `seqinfo.ini` sidecar files that ship with MOTChallenge sequences.
- [`Accumulators`][norfair.metrics.Accumulators] — a thin wrapper around
  `motmetrics` accumulators that lets you evaluate a whole dataset and print
  a summary table.

## Typical workflow

1. For each sequence in the dataset, use `DetectionFileParser` to replay the
   provided detections through your `Tracker`, and write the output with
   `PredictionsTextFile`.
2. Once all sequences have been processed, compare predictions against
   ground truth with `Accumulators` to get the standard MOT metrics.

See the
[MOTChallenge demo](https://github.com/tryolabs/norfair/tree/master/demos/motchallenge)
for a complete script that exercises all of the above.

## API

::: norfair.metrics
    options:
        show_root_heading: false
        show_if_no_docstring: true

## See also

- [Tracker](tracker.md) — the tracker being evaluated.
- [Distances](distances.md) — the distance / threshold combination you pick
  here directly drives your MOT scores.
