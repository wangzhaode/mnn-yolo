# Usage

## Compile MNN library
### Linx/Mac
```bash
git clone https://github.com/alibaba/MNN.git
cd MNN
# copy header file
cp -r include /path/to/MNNExample/mobilenet/cpp
cp -r tools/cv/include /path/to/MNNExample/yolov8/cpp
mkdir build
cmake -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON ..
make -j8
cp libMNN.so express/libMNN_Express.so tools/cv/libMNNOpenCV.so /path/to/MNNExample/yolov8/cpp/libs
```

### Windows
```bash
# Visual Studio xxxx Developer Command Prompt
powershell
git clone https://github.com/alibaba/MNN.git
cd MNN
# copy header file
cp -r include /path/to/MNNExample/mobilenet/cpp
cp -r tools/cv/include /path/to/MNNExample/yolov8/cpp
mkdir build
cmake -G "Ninja" -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON ..
ninja
cp MNN.dll MNN.lib /path/to/MNNExample/yolov8/cpp/build
```

## Build and Run

#### Linux/Mac
```bash
mkdir build && cd build
cmake ..
make -j4
./yolov8_demo yolov8n.mnn test.jpg
```
#### Windows
```bash
# Visual Studio xxxx Developer Command Prompt
powershell
mkdir build && cd build
cmake -G "Ninja" ..
ninja
./yolov8_demo yolov8n.mnn test.jpg
```