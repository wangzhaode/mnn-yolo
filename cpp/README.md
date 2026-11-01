# Usage

## Compile MNN library

### Linux/Mac
```bash
git clone https://github.com/alibaba/MNN.git
cd MNN
mkdir build && cd build
cmake -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON ..
make -j8

# copy headers and libs into mnn-yolo/cpp
cp -r ../include       /path/to/mnn-yolo/cpp/include/MNN
cp -r ../tools/cv/include/cv /path/to/mnn-yolo/cpp/include/cv
cp libMNN.so express/libMNN_Express.so tools/cv/libMNNOpenCV.so /path/to/mnn-yolo/cpp/libs/
```

### Windows
```bash
# Visual Studio xxxx Developer Command Prompt -> powershell
git clone https://github.com/alibaba/MNN.git
cd MNN
mkdir build && cd build
cmake -G "Ninja" -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON ..
ninja

cp -r ../include       /path/to/mnn-yolo/cpp/include/MNN
cp -r ../tools/cv/include/cv /path/to/mnn-yolo/cpp/include/cv
cp MNN.dll MNN.lib /path/to/mnn-yolo/cpp/libs/
```

## Build and Run

### Linux/Mac
```bash
mkdir build && cd build
cmake ..
make -j4
./yolo11_demo yolo11n.mnn ../../resource/test.jpg
```

### Windows
```bash
# Visual Studio xxxx Developer Command Prompt -> powershell
mkdir build && cd build
cmake -G "Ninja" ..
ninja
./yolo11_demo yolo11n.mnn ../../resource/test.jpg
```

## Available demos

| demo | output format |
|------|---------------|
| `yolov5_demo`   | `[N, 85]` (cx, cy, w, h, box_conf, prob×80) |
| `yolov6_demo`   | `[boxes, classes, scores, num]` (with builtin NMS) |
| `yolov7_demo`   | `[N, 7]` (batch_id, x0, y0, x1, y1, class_id, score) |
| `yolov8_demo`   | `[84, 8400]` (cx, cy, w, h, prob×80) |
| `yolov9_demo`   | same as yolov8 |
| `yolov10_demo`  | `[1, 300, 6]` (x0, y0, x1, y1, score, class_id) |
| `yolo11_demo`   | same as yolov8 |
| `yolo12_demo`   | same as yolov8 |
| `yolo13_demo`   | same as yolov8 |
| `yoloe_demo`    | same as yolov8 (open-vocabulary, classes baked in via `set_classes`) |
| `yolo11_seg_demo`  | det branch + 32-channel mask proto |
| `yolo11_pose_demo` | `[56, 8400]` (4 box + 1 conf + 17×3 keypoints) |
| `yolo11_obb_demo`  | `[20, 8400]` (4 box + 15 classes + 1 angle) |

Usage: `./<demo> model.mnn input.jpg [forwardType] [precision] [thread]`

- forwardType: 0=CPU, 3=OPENCL, 7=METAL, 9=CUDA, 13=NN
- precision: 0=normal, 1=high, 2=low
