#-- coding:utf8 --
# YOLO11-seg (instance segmentation)
# Two outputs:
#   output0: [1, 116, 8400] = 4 box + 80 class probs + 32 mask coefficients
#   output1: [1, 32, 160, 160] = mask prototypes
# Per-instance mask is computed as: sigmoid(coef @ proto.reshape(32, -1))
# then resized to the input size and cropped to the bbox.
import argparse

import MNN
import MNN.numpy as np
import MNN.cv as cv2

COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def inference(model, img, precision, backend, thread):
    config = {'precision': precision, 'backend': backend, 'numThread': thread}
    rt = MNN.nn.create_runtime_manager((config,))
    # Multi-output model: passing explicit output names is required, otherwise
    # onForward segfaults on this build of MNN.
    net = MNN.nn.load_module_from_file(model, ['images'], ['output0', 'output1'], runtime_manager=rt)
    original_image = cv2.imread(img)
    ih, iw, _ = original_image.shape
    length = max((ih, iw))
    scale = length / 640
    image = np.pad(original_image, [[0, length - ih], [0, length - iw], [0, 0]], 'constant')
    image = cv2.resize(image, (640, 640), 0., 0., cv2.INTER_LINEAR, -1, [0., 0., 0.], [1./255., 1./255., 1./255.])
    input_var = np.expand_dims(image, 0)
    input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
    outputs = net.onForward(input_var)
    # outputs[0]: [1, 116, 8400] det branch; outputs[1]: [1, 32, 160, 160] proto
    det = MNN.expr.convert(outputs[0], MNN.expr.NCHW).squeeze()
    proto = MNN.expr.convert(outputs[1], MNN.expr.NCHW).squeeze()

    cx = det[0]
    cy = det[1]
    w = det[2]
    h = det[3]
    cls_probs = det[4:84]
    mask_coefs = det[84:]  # [32, 8400]

    x0 = cx - w * 0.5
    y0 = cy - h * 0.5
    x1 = cx + w * 0.5
    y1 = cy + h * 0.5
    boxes = np.stack([x0, y0, x1, y1], axis=1)
    scores = np.max(cls_probs, 0)
    class_ids = np.argmax(cls_probs, 0)
    result_ids = MNN.expr.nms(boxes, scores, 100, 0.45, 0.25)
    result_boxes = boxes[result_ids]
    result_scores = scores[result_ids]
    result_class_ids = class_ids[result_ids]
    # kept mask coefs: [N, 32]
    kept_coefs = mask_coefs.transpose([1, 0])[result_ids]
    proto_flat = proto.reshape([32, 160 * 160])

    for i in range(len(result_boxes)):
        x0, y0, x1, y1 = result_boxes[i].read_as_tuple()
        bx0 = int(max(x0 * scale, 0))
        by0 = int(max(y0 * scale, 0))
        bx1 = int(min(x1 * scale, original_image.shape[1] - 1))
        by1 = int(min(y1 * scale, original_image.shape[0] - 1))
        class_idx = int(result_class_ids[i])
        score = float(result_scores[i])
        coef = kept_coefs[i].reshape([1, 32])
        mask_logits = np.matmul(coef, proto_flat).reshape([160, 160])
        # report mean activation as a sanity check; full pixel-overlay is left to user
        mean_act = float(np.mean(mask_logits))
        label = f'{COCO_NAMES[class_idx]} {score:.2f}'
        print(f'### box: [{bx0}, {by0}, {bx1}, {by1}], class: {label}, mask_mean: {mean_act:.3f}')
        cv2.rectangle(original_image, (bx0, by0), (bx1, by1), (0, 0, 255), 2)
    cv2.imwrite('res.jpg', original_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='the yolo11-seg mnn model path')
    parser.add_argument('--img', type=str, required=True, help='the input image path')
    parser.add_argument('--precision', type=str, default='normal')
    parser.add_argument('--backend', type=str, default='CPU')
    parser.add_argument('--thread', type=int, default=4)
    args = parser.parse_args()
    inference(args.model, args.img, args.precision, args.backend, args.thread)
