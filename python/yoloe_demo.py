#-- coding:utf8 --
# YOLOE (Real-Time Seeing Anything, https://github.com/THU-MIG/yoloe)
# Open-vocabulary detector. To export an MNN model with custom classes:
#   from ultralytics import YOLOE
#   model = YOLOE("yoloe-v8s-seg.pt")
#   names = ["person", "bus"]
#   model.set_classes(names, model.get_text_pe(names))
#   model.export(format="mnn")
# After export, the model behaves like a regular detector: output shape
#   [4 + len(names), 8400] -> 4 box (cx, cy, w, h) + N class probs.
# Pass the same names list to this script with --names so labels match.
import argparse

import MNN
import MNN.numpy as np
import MNN.cv as cv2

def inference(model, img, names, precision, backend, thread):
    config = {'precision': precision, 'backend': backend, 'numThread': thread}
    rt = MNN.nn.create_runtime_manager((config,))
    net = MNN.nn.load_module_from_file(model, [], [], runtime_manager=rt)
    original_image = cv2.imread(img)
    ih, iw, _ = original_image.shape
    length = max((ih, iw))
    scale = length / 640
    image = np.pad(original_image, [[0, length - ih], [0, length - iw], [0, 0]], 'constant')
    image = cv2.resize(image, (640, 640), 0., 0., cv2.INTER_LINEAR, -1, [0., 0., 0.], [1./255., 1./255., 1./255.])
    input_var = np.expand_dims(image, 0)
    input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
    output_var = net.forward(input_var)
    output_var = MNN.expr.convert(output_var, MNN.expr.NCHW)
    output_var = output_var.squeeze()
    # output_var shape: [4 + N, 8400]; first 4 are cx, cy, w, h; rest are per-class probs
    cx = output_var[0]
    cy = output_var[1]
    w = output_var[2]
    h = output_var[3]
    probs = output_var[4:]
    x0 = cx - w * 0.5
    y0 = cy - h * 0.5
    x1 = cx + w * 0.5
    y1 = cy + h * 0.5
    boxes = np.stack([x0, y0, x1, y1], axis=1)
    scores = np.max(probs, 0)
    class_ids = np.argmax(probs, 0)
    result_ids = MNN.expr.nms(boxes, scores, 100, 0.45, 0.25)
    result_boxes = boxes[result_ids]
    result_scores = scores[result_ids]
    result_class_ids = class_ids[result_ids]
    for i in range(len(result_boxes)):
        x0, y0, x1, y1 = result_boxes[i].read_as_tuple()
        x0, y0, x1, y1 = int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale)
        class_idx = int(result_class_ids[i])
        score = float(result_scores[i])
        cls_name = names[class_idx] if class_idx < len(names) else f'cls_{class_idx}'
        label = f'{cls_name} {score:.2f}'
        print(f'### box: [{x0}, {y0}, {x1}, {y1}], class: {label}')
        cv2.rectangle(original_image, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite('res.jpg', original_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='the yoloe mnn model path')
    parser.add_argument('--img', type=str, required=True, help='the input image path')
    parser.add_argument('--names', type=str, required=True,
                        help='comma-separated class names used during export (e.g. "person,bus")')
    parser.add_argument('--precision', type=str, default='normal', help='inference precision: normal, low, high, lowBF')
    parser.add_argument('--backend', type=str, default='CPU', help='inference backend: CPU, OPENCL, OPENGL, NN, VULKAN, METAL, TRT, CUDA, HIAI')
    parser.add_argument('--thread', type=int, default=4, help='inference using thread: int')
    args = parser.parse_args()
    names = [n.strip() for n in args.names.split(',')]
    inference(args.model, args.img, names, args.precision, args.backend, args.thread)
