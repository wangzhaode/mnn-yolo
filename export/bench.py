from ultralytics.utils.benchmarks import benchmark


benchmark(model="yolo11n.onnx", data="coco8.yaml", imgsz=640, half=False)
