# Ultralytics YOLO-Pose Example

Track human pose keypoints in video using [Ultralytics YOLO-Pose](https://docs.ultralytics.com/tasks/pose/) and Norfair.

## Instructions

1. Build and run the Docker container with `./run_gpu.sh`, or install dependencies locally:

   ```bash
   pip install -r requirements.txt
   ```

2. Run on a video file:

   ```bash
   python src/demo.py video.mp4
   ```

For additional settings, run `python src/demo.py --help`.

## Explanation

This demo uses a YOLO-Pose model to detect 17 COCO keypoints per person, converts them to Norfair `Detection` objects, and tracks them across frames using normalized mean Euclidean distance. Tracked keypoints are drawn on each frame.
