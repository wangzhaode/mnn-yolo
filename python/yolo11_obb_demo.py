#-- coding:utf8 --
# YOLO11-obb (Oriented Bounding Box, DOTA-v1, 15 classes)
# Output: [1, 20, 8400] -> 4 box (cx, cy, w, h) + 15 class probs + 1 angle (rad)
import argparse
import math

import MNN
import MNN.numpy as np
import MNN.cv as cv2

DOTA_NAMES = [
    'plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
    'basketball court', 'ground track field', 'harbor', 'bridge',
    'large vehicle', 'small vehicle', 'helicopter', 'roundabout',
    'soccer ball field', 'swimming pool',
]

def inference(model, img, precision, backend, thread):
    config = {'precision': precision, 'backend': backend, 'numThread': thread}
    rt = MNN.nn.create_runtime_manager((config,))
    net = MNN.nn.load_module_from_file(model, [], [], runtime_manager=rt)
    original_image = cv2.imread(img)
    ih, iw, _ = original_image.shape
    length = max((ih, iw))
    scale = length / 1024  # OBB models default to 1024x1024
    image = np.pad(original_image, [[0, length - ih], [0, length - iw], [0, 0]], 'constant')
    image = cv2.resize(image, (1024, 1024), 0., 0., cv2.INTER_LINEAR, -1, [0., 0., 0.], [1./255., 1./255., 1./255.])
    input_var = np.expand_dims(image, 0)
    input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
    output_var = net.forward(input_var)
    output_var = MNN.expr.convert(output_var, MNN.expr.NCHW)
    output_var = output_var.squeeze()
    # output shape: [20, anchors]
    cx = output_var[0]
    cy = output_var[1]
    w = output_var[2]
    h = output_var[3]
    cls_probs = output_var[4:19]
    angle = output_var[19]

    # use axis-aligned bbox for NMS
    x0 = cx - w * 0.5
    y0 = cy - h * 0.5
    x1 = cx + w * 0.5
    y1 = cy + h * 0.5
    aabb = np.stack([x0, y0, x1, y1], axis=1)
    scores = np.max(cls_probs, 0)
    class_ids = np.argmax(cls_probs, 0)
    result_ids = MNN.expr.nms(aabb, scores, 100, 0.45, 0.25)
    if result_ids.shape[0] == 0:
        print('### no obb detections')
        cv2.imwrite('res.jpg', original_image)
        return

    cx_t = cx.read_as_tuple()
    cy_t = cy.read_as_tuple()
    w_t = w.read_as_tuple()
    h_t = h.read_as_tuple()
    angle_t = angle.read_as_tuple()
    result_t = result_ids.read_as_tuple()
    scores_t = scores.read_as_tuple()
    cls_t = class_ids.read_as_tuple()

    for r in result_t:
        if r < 0:
            break
        rcx, rcy = cx_t[r] * scale, cy_t[r] * scale
        rw, rh = w_t[r] * scale, h_t[r] * scale
        ang = angle_t[r]
        cls_idx = int(cls_t[r])
        score = float(scores_t[r])
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        hw, hh = rw * 0.5, rh * 0.5
        corners = []
        for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            px = int(rcx + dx * cos_a - dy * sin_a)
            py = int(rcy + dx * sin_a + dy * cos_a)
            corners.append((px, py))
        for i in range(4):
            cv2.line(original_image, corners[i], corners[(i + 1) % 4], (0, 0, 255), 2)
        label = f'{DOTA_NAMES[cls_idx]} {score:.2f}'
        print(f'### obb: cx={rcx:.1f} cy={rcy:.1f} w={rw:.1f} h={rh:.1f} angle={ang:.3f}rad, class: {label}')
    cv2.imwrite('res.jpg', original_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='the yolo11-obb mnn model path')
    parser.add_argument('--img', type=str, required=True, help='the input image path')
    parser.add_argument('--precision', type=str, default='normal')
    parser.add_argument('--backend', type=str, default='CPU')
    parser.add_argument('--thread', type=int, default=4)
    args = parser.parse_args()
    inference(args.model, args.img, args.precision, args.backend, args.thread)
