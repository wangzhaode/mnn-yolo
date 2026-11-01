# mnn-yolo models

MNN-format YOLO models for the [mnn-yolo examples](https://github.com/wangzhaode/mnn-yolo).
These files are converted model weights, not models trained by this project.
The files have different upstream authors and licenses; consult the relevant
upstream project and checkpoint terms before using or redistributing them.
In particular, Ultralytics YOLO26 is distributed under AGPL-3.0 or an
Ultralytics enterprise license; see [Ultralytics licensing](https://www.ultralytics.com/license).

## YOLO26n

These fixed-shape, batch-one models were exported from official Ultralytics
checkpoints through ONNX and converted with MNN 3.6.0. They use float32 model
weights and require a compatible MNN runtime.

| File | Task | Input size (height × width) | Output layout |
| --- | --- | --- | --- |
| `yolo26n.mnn` | Object detection | 640 × 640 | end-to-end boxes |
| `yolo26n-seg.mnn` | Instance segmentation | 640 × 640 | end-to-end boxes + prototypes |
| `yolo26n-sem.mnn` | Semantic segmentation | 1024 × 2048 | per-pixel class IDs |
| `yolo26n-pose.mnn` | Pose estimation | 640 × 640 | end-to-end boxes + 17 keypoints |
| `yolo26n-obb.mnn` | Oriented boxes | 1024 × 1024 | end-to-end rotated boxes |
| `yolo26n-cls.mnn` | Image classification | 224 × 224 | class scores |
| `yolo26n-depth.mnn` | Depth estimation | 768 × 768 | per-pixel depth |

The corresponding Python examples and preprocessing details are in the
[project guide](https://github.com/wangzhaode/mnn-yolo/tree/main/python).
The YOLO26 checkpoints originate from [Ultralytics](https://github.com/ultralytics/ultralytics).
Validate a model on your target MNN version and data before deployment.

## Earlier models

The YOLOv5–YOLO11 detection files reproduce the original mnn-yolo release
assets. They have legacy output layouts and runtime requirements; use the
matching standalone examples in the project. Source projects include
[YOLOv5](https://github.com/ultralytics/yolov5),
[YOLOv6](https://github.com/meituan/YOLOv6),
[YOLOv7](https://github.com/WongKinYiu/yolov7),
[YOLOv8](https://github.com/ultralytics/ultralytics),
[YOLOv9](https://github.com/WongKinYiu/yolov9) and
[YOLOv10](https://github.com/THU-MIG/yolov10).

## Download

```bash
pip install modelscope
modelscope download --model zhaode/mnn-yolo yolo26n.mnn --local_dir ./models
```

See `SHA256SUMS` for file integrity checks.
