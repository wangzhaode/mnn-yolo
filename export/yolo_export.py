import sys
from ultralytics import YOLO
from MNN.tools.mnnconvert import Tools

model_path = sys.argv[1]

# Load a model
model = YOLO(model_path)

# Export the model to ONNX format
onnx_path = model.export(format="onnx")

# Convert ONNX to MNN
mnn_path = onnx_path.replace('.onnx', '.mnn')
convert_args = [
            '',
            '-f',
            'ONNX',
            '--modelFile',
            str(onnx_path),
            '--MNNModel',
            str(mnn_path),
            '--fp16'
        ]
Tools.mnnconvert(convert_args)