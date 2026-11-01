# Python demos

## Task-oriented demo: YOLOv8, YOLO11 and YOLO26

`yolo_tasks.py` uses MNN's Python Interpreter for inference and NumPy/OpenCV
for preprocessing, postprocessing and visualization. Install `MNN`, `numpy`,
`opencv-python` and `Pillow`. It accepts a fixed batch-one NCHW image input. Image
dimensions come from the exported model, rather than a hardcoded 640/1024.
Only the CPU backend is currently exposed.

Export with a compatible Ultralytics package and `MNNConvert` on PATH:

```bash
pip install ultralytics onnx
python export/export_ultralytics.py yolo26n.pt --imgsz 640
python export/export_ultralytics.py yolo26n-seg.pt --imgsz 640
python export/export_ultralytics.py yolo26n-pose.pt --imgsz 640
python export/export_ultralytics.py yolo26n-obb.pt --imgsz 1024
python export/export_ultralytics.py yolo26n-cls.pt --imgsz 224
python export/export_ultralytics.py yolo11n-seg.pt --imgsz 640
```

Alternatively, download the preconverted YOLO26n models from
[ModelScope](https://modelscope.cn/models/zhaode/mnn-yolo). See its model card
for the fixed input size of each file:

```bash
pip install modelscope
modelscope download --model zhaode/mnn-yolo yolo26n-seg.mnn --local_dir ./models
python python/yolo_tasks.py --model models/yolo26n-seg.mnn --img resource/test.jpg --task segment
```

Run from the repository root (model paths depend on where export wrote them):

```bash
python python/yolo_tasks.py --model yolo26n.mnn --img resource/test.jpg --task detect
python python/yolo_tasks.py --model yolo26n-seg.mnn --img resource/test.jpg --task segment
python python/yolo_tasks.py --model yolo26n-pose.mnn --img resource/test.jpg --task pose
python python/yolo_tasks.py --model yolo26n-obb.mnn --img resource/test.jpg --task obb
python python/yolo_tasks.py --model yolo26n-cls.mnn --img resource/test.jpg --task classify
python python/yolo_tasks.py --model yolo11n-seg.mnn --img resource/test.jpg --task segment --layout raw
```

The default `--layout end2end` expects YOLO26-style `[1,N,6+extra]` outputs.
For raw YOLOv8/YOLO11 outputs use `--layout raw`; these expect
`[1,4+classes+extra,N]` with xywh boxes. Raw defaults are 80 classes for
detection/segmentation, one for pose and 15 for OBB; override with `--classes`
for custom checkpoints. Pose assumes three values per keypoint and defaults
to 17 keypoints (`--keypoints` changes the count). OBB raw inference uses
class-aware rotated IoU NMS; it may differ from Ultralytics probabilistic NMS.
End-to-end exports already select detections and do not run additional NMS.
Classification prints class IDs, not ImageNet labels. Model-specific or
promptable YOLOE exports may have different input/output contracts and are not
covered by this entry point.

The result image defaults to `res.jpg`; `result.npz` holds scores, model-space
predictions and the letterbox transform, instance masks, semantic IDs, or
depth values. Override the destinations with `--output` and `--data`.
Detection rows are `[x1,y1,x2,y2,confidence,class,...]`, except OBB rows use
`[cx,cy,w,h,confidence,class,angle]`; coordinates are in model-input pixels.

YOLO26 semantic and depth paths accept respectively logits
`[1,C,H,W]` or class IDs `[1,H,W]`, and depth `[1,1,H,W]` or `[1,H,W]`.
Semantic removes letterbox padding; depth uses direct resize. They need a
recent Ultralytics version with those tasks and appropriate model weights:

```bash
python export/export_ultralytics.py yolo26n-sem.pt --imgsz 1024 2048
python export/export_ultralytics.py yolo26n-depth.pt --imgsz 768
python python/yolo_tasks.py --model yolo26n-sem.mnn --img street.jpg --task semantic
python python/yolo_tasks.py --model yolo26n-depth.mnn --img street.jpg --task depth
```

Depth visualization is normalized per image; use the float map in the NPZ for
the model's values. Semantic class names depend on the training dataset.
On MNN 3.6.0 and Ultralytics 8.4.194, all seven official YOLO26n checkpoints
were converted and run on CPU. On the sample images, YOLO26 instance masks
matched the Ultralytics ONNX results with IoU above 0.999; the semantic class
map matched exactly; the depth map differed by 0.000204 on average; the OBB
example detected one object on a DOTA8 image, matching the ONNX result; and
classification's top five IDs matched. These are single-image checks, not
dataset accuracy measurements.
YOLO11 instance masks were also compared with Ultralytics on `resource/test.jpg`.

Run the postprocessing tests with:

```bash
python -m unittest discover -s python -p 'test_yolo_tasks.py' -v
```

## Install MNN
```bash
pip install MNN
```

## Run

Each model has its own `*_demo.py`. Output goes to `res.jpg`.

### Detection

```bash
python yolov5_demo.py  --model yolov5s.mnn  --img ../resource/test.jpg
python yolov6_demo.py  --model yolov6n.mnn  --img ../resource/test.jpg
python yolov7_demo.py  --model yolov7.mnn   --img ../resource/test.jpg
python yolov8_demo.py  --model yolov8n.mnn  --img ../resource/test.jpg
python yolov9_demo.py  --model yolov9c.mnn  --img ../resource/test.jpg
python yolov10_demo.py --model yolov10n.mnn --img ../resource/test.jpg
python yolo11_demo.py  --model yolo11n.mnn  --img ../resource/test.jpg
python yolo12_demo.py  --model yolo12n.mnn  --img ../resource/test.jpg
python yolo13_demo.py  --model yolo13n.mnn  --img ../resource/test.jpg
```

### YOLOE (open-vocabulary)
Pass the same names you used at export time:

```bash
python yoloe_demo.py --model yoloe.mnn --img ../resource/test.jpg --names "person,bus"
```

### YOLO11 task variants

These older standalone demos are retained for reference. The segmentation
demo reports mask statistics and draws boxes only; use `yolo_tasks.py` above
to generate and visualize pixel masks.

```bash
python yolo11_seg_demo.py  --model yolo11n-seg.mnn  --img ../resource/test.jpg
python yolo11_pose_demo.py --model yolo11n-pose.mnn --img ../resource/test.jpg
python yolo11_obb_demo.py  --model yolo11n-obb.mnn  --img ../resource/test.jpg
```

## Common flags

| flag | default | description |
|------|---------|-------------|
| `--precision` | `normal` | `normal`, `low`, `high`, `lowBF` |
| `--backend`   | `CPU`    | `CPU`, `OPENCL`, `OPENGL`, `NN`, `VULKAN`, `METAL`, `TRT`, `CUDA`, `HIAI` |
| `--thread`    | `4`      | inference thread count |
