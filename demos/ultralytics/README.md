# Ultralytics YOLO Example

Object detection and tracking using [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO models (YOLOv8, YOLO11, etc.) with Norfair Enough.

## Instructions

1. Build and run the Docker container:

   ```bash
   docker build -t norfair-ultralytics .
   docker run -it --rm -v $(pwd)/src:/demo/src norfair-ultralytics bash
   ```

2. Copy a video to the `src` folder.

3. Run the demo:

   ```bash
   python demo.py video.mp4
   ```

### Options

```
--model          Model name (default: yolo11n.pt)
--conf-threshold Confidence threshold (default: 0.25)
--device         Inference device: cpu, cuda, mps (default: auto)
--track-points   Track by 'centroid' or 'bbox' (default: bbox)
--classes        Filter by class ID: --classes 0 2 3
```

For all options, run `python demo.py --help`.
