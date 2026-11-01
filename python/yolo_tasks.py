#!/usr/bin/env python3
"""Run batch-one Ultralytics ONNX-to-MNN exports with explicit task/layout.

Uses ordinary NumPy/OpenCV for postprocessing, and MNN only for inference.
See python/README.md for supported export contracts and limitations.
"""
import argparse

import cv2
import numpy as np


def preprocess(image, height, width, task):
    h, w = image.shape[:2]
    if task == "classify":
        from PIL import Image
        # Match Ultralytics' evaluation transforms: PIL bilinear resize of the
        # shortest edge, then center crop. PIL antialiasing matters here.
        if height == width:
            nh, nw = (height, int(width * w / h)) if h <= w else (int(height * h / w), width)
        else:
            nh, nw = height, width
        resized = np.asarray(Image.fromarray(image).resize((nw, nh), Image.Resampling.BILINEAR))
        top, left = round((nh - height) / 2), round((nw - width) / 2)
        canvas = resized[top:top + height, left:left + width]
        transform = (1.0, 0, 0)
    elif task == "depth":
        canvas = cv2.resize(image, (width, height))
        transform = (1.0, 0, 0)
    else:
        ratio = min(width / w, height / h)
        nw, nh = round(w * ratio), round(h * ratio)
        left, top = (width - nw) // 2, (height - nh) // 2
        canvas = np.full((height, width, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
        transform = (ratio, left, top)
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor), transform


def run_mnn(path, image, task, threads):
    import MNN
    interpreter = MNN.Interpreter(str(path))
    session = interpreter.createSession({"backend": "CPU", "numThread": threads})
    inputs = interpreter.getSessionInputAll(session)
    if len(inputs) != 1:
        raise ValueError(f"expected one image input, found {list(inputs)}")
    tensor = next(iter(inputs.values()))
    shape = tensor.getShape()
    if len(shape) != 4 or shape[0:2] != (1, 3) or min(shape[2:]) <= 0:
        raise ValueError(f"expected fixed NCHW input [1,3,H,W], got {shape}")
    data, transform = preprocess(image, shape[2], shape[3], task)
    host = MNN.Tensor(shape, MNN.Halide_Type_Float, data, MNN.Tensor_DimensionType_Caffe)
    tensor.copyFromHostTensor(host)
    interpreter.runSession(session)
    outputs = {name: value.getNumpyData().copy() for name, value in interpreter.getSessionOutputAll(session).items()}
    return outputs, transform, shape[2:]


def xywh_to_xyxy(boxes):
    out = boxes.copy()
    out[:, :2] = boxes[:, :2] - boxes[:, 2:4] / 2
    out[:, 2:4] = boxes[:, :2] + boxes[:, 2:4] / 2
    return out


def select_predictions(pred, task, layout, nc, extra, conf, iou):
    """Return rows xyxy/xywh, confidence, class, extras in model coordinates."""
    a = np.asarray(pred)
    if a.ndim != 3 or a.shape[0] != 1:
        raise ValueError(f"expected batch-one prediction tensor, got {a.shape}")
    a = a[0]
    if layout == "end2end":
        if a.shape[1] != 6 + extra:
            raise ValueError(f"end2end {task}: expected {6 + extra} columns, got {a.shape}")
        return a[a[:, 4] >= conf].copy()
    if a.shape[0] != 4 + nc + extra:
        raise ValueError(f"raw {task}: expected {4 + nc + extra} channels, got {a.shape}")
    a = a.T
    cls = a[:, 4:4 + nc]
    ids = cls.argmax(axis=1)
    scores = cls[np.arange(len(a)), ids]
    good = scores >= conf
    a, scores, ids = a[good], scores[good], ids[good]
    boxes = a[:, :4] if task == "obb" else xywh_to_xyxy(a[:, :4])
    kept = []
    for cls_id in np.unique(ids):
        inds = np.flatnonzero(ids == cls_id)
        if task == "obb":
            rects = [((float(b[0]), float(b[1])), (float(b[2]), float(b[3])),
                      float(a[i, -1] * 180 / np.pi)) for i, b in zip(inds, boxes[inds])]
            chosen = cv2.dnn.NMSBoxesRotated(rects, scores[inds].tolist(), conf, iou)
        else:
            b = boxes[inds].copy()
            b[:, 2:4] -= b[:, :2]
            chosen = cv2.dnn.NMSBoxes(b.tolist(), scores[inds].tolist(), conf, iou)
        kept.extend(inds[np.asarray(chosen, dtype=int).reshape(-1)].tolist())
    kept = sorted(kept, key=lambda i: scores[i], reverse=True)[:300]
    return np.concatenate((boxes[kept], scores[kept, None], ids[kept, None], a[kept, 4 + nc:]), axis=1)


def restore_map(data, image_shape, input_shape, transform, nearest=False):
    h, w = image_shape[:2]
    ih, iw = input_shape
    ratio, left, top = transform
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    data = cv2.resize(data, (iw, ih), interpolation=interpolation)
    nh, nw = round(h * ratio), round(w * ratio)
    return cv2.resize(data[top:top + nh, left:left + nw], (w, h), interpolation=interpolation)


def instance_masks(rows, proto, image_shape, input_shape, transform):
    p = np.asarray(proto)
    if p.ndim != 4 or p.shape[0] != 1 or rows.shape[1] != 6 + p.shape[1]:
        raise ValueError(f"unexpected prototype/prediction shapes: {p.shape}, {rows.shape}")
    h, w = image_shape[:2]
    masks = []
    ratio, left, top = transform
    for row in rows:
        logits = (row[6:] @ p[0].reshape(p.shape[1], -1)).reshape(p.shape[2:])
        # Bilinear interpolation of logits before thresholding matches sigmoid > 0.5.
        mask = restore_map(logits, image_shape, input_shape, transform) > 0
        x1, y1, x2, y2 = row[:4]
        x1, x2 = np.clip([(x1 - left) / ratio, (x2 - left) / ratio], 0, w)
        y1, y2 = np.clip([(y1 - top) / ratio, (y2 - top) / ratio], 0, h)
        yy, xx = np.ogrid[:h, :w]
        masks.append(mask & (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2))
    return np.stack(masks) if masks else np.zeros((0, h, w), dtype=bool)


def resize_align_corners(data, shape):
    """Bilinear resize with the align_corners=True convention used by depth."""
    h, w = shape
    ys = np.linspace(0, data.shape[0] - 1, h)
    xs = np.linspace(0, data.shape[1] - 1, w)
    y0, x0 = ys.astype(int), xs.astype(int)
    y1, x1 = np.minimum(y0 + 1, data.shape[0] - 1), np.minimum(x0 + 1, data.shape[1] - 1)
    wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]
    return ((1 - wy) * ((1 - wx) * data[y0[:, None], x0] + wx * data[y0[:, None], x1]) +
            wy * ((1 - wx) * data[y1[:, None], x0] + wx * data[y1[:, None], x1])).astype(np.float32)


def dense_output(output, task, image_shape, input_shape, transform):
    a = np.asarray(output)
    if a.shape[0] != 1:
        raise ValueError(f"expected batch one, got {a.shape}")
    a = a[0]
    h, w = image_shape[:2]
    if task == "depth":
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 2:
            raise ValueError(f"expected depth [1,1,H,W] or [1,H,W], got {output.shape}")
        return resize_align_corners(resize_align_corners(a, input_shape), (h, w))
    if a.ndim == 2:
        return restore_map(a.astype(np.float32), image_shape, input_shape, transform, nearest=True).astype(np.int32)
    if a.ndim != 3:
        raise ValueError(f"expected semantic logits [1,C,H,W] or class map [1,H,W], got {output.shape}")
    channels = np.stack([restore_map(ch, image_shape, input_shape, transform) for ch in a])
    return (channels[0] > 0).astype(np.int32) if len(channels) == 1 else channels.argmax(axis=0).astype(np.int32)


def color_for(i):
    return tuple(int(x) for x in ((37 * i + 73) % 256, (17 * i + 149) % 256, (97 * i + 53) % 256))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--img', required=True)
    parser.add_argument('--task', required=True, choices=['detect', 'segment', 'pose', 'obb', 'classify', 'semantic', 'depth'])
    parser.add_argument('--layout', choices=['raw', 'end2end'], default='end2end', help='prediction layout; YOLO26 defaults to end2end, YOLOv8/11 to raw')
    parser.add_argument('--classes', type=int, default=None, help='number of classes for raw exports (defaults: 80 detect/segment, 1 pose, 15 OBB)')
    parser.add_argument('--keypoints', type=int, default=17, help='pose keypoints, each with x/y/confidence')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', default='res.jpg')
    parser.add_argument('--data', default='result.npz', help='machine-readable results')
    args = parser.parse_args()
    image = cv2.imread(args.img)
    if image is None:
        raise ValueError(f'cannot read image: {args.img}')
    outputs, transform, shape = run_mnn(args.model, image, args.task, args.threads)
    print('MNN outputs:', {k: v.shape for k, v in outputs.items()})
    values = list(outputs.values())
    result = {}
    if args.task in ('semantic', 'depth'):
        if len(values) != 1:
            raise ValueError('expected one dense output')
        dense = dense_output(values[0], args.task, image.shape, shape, transform)
        result[args.task] = dense
        if args.task == 'semantic':
            colors = np.array([color_for(i) for i in range(int(dense.max()) + 1)], dtype=np.uint8)
            image = cv2.addWeighted(image, .5, colors[dense], .5, 0)
        else:
            lo, hi = np.percentile(dense, [2, 98])
            vis = np.clip((dense - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            image = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    elif args.task == 'classify':
        if len(values) != 1 or values[0].ndim != 2 or values[0].shape[0] != 1:
            raise ValueError('expected classification output [1, classes]')
        scores = values[0][0]
        result['scores'] = scores
        for i in scores.argsort()[::-1][:5]:
            print(f'class {i}: {scores[i]:.4f}')
    else:
        protos = [v for v in values if v.ndim == 4]
        preds = [v for v in values if v.ndim == 3]
        if len(preds) != 1 or (args.task == 'segment' and len(protos) != 1):
            raise ValueError('expected one prediction tensor and, for segmentation, one prototype tensor')
        extra = protos[0].shape[1] if args.task == 'segment' else args.keypoints * 3 if args.task == 'pose' else 1 if args.task == 'obb' else 0
        nc = args.classes or {'detect': 80, 'segment': 80, 'pose': 1, 'obb': 15}[args.task]
        rows = select_predictions(preds[0], args.task, args.layout, nc, extra, args.conf, args.iou)
        ratio, left, top = transform
        if args.task == 'segment':
            masks = instance_masks(rows, protos[0], image.shape, shape, transform)
            result['masks'] = masks
            for i, mask in enumerate(masks):
                image[mask] = (image[mask].astype(np.float32) * .5 + np.array(color_for(i)) * .5).astype(np.uint8)
        result['predictions_model_coords'] = rows
        for row in rows:
            if args.task == 'obb':
                cx, cy, w, h = row[:4]
                rect = (((cx - left) / ratio, (cy - top) / ratio), (w / ratio, h / ratio), row[6] * 180 / np.pi)
                cv2.polylines(image, [cv2.boxPoints(rect).astype(np.int32)], True, (0, 0, 255), 2)
            else:
                box = row[:4].copy()
                box[[0, 2]] = (box[[0, 2]] - left) / ratio
                box[[1, 3]] = (box[[1, 3]] - top) / ratio
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                if args.task == 'pose':
                    for x, y, score in row[6:].reshape(-1, 3):
                        if score >= .5:
                            cv2.circle(image, (int((x - left) / ratio), int((y - top) / ratio)), 3, (0, 255, 0), -1)
            print(f'class {int(row[5])} score {row[4]:.3f} box {row[:4].round(1)}')
        result['transform'] = np.asarray(transform)
        result['input_shape'] = np.asarray(shape)
    if not cv2.imwrite(args.output, image):
        raise RuntimeError(f'cannot write {args.output}')
    np.savez_compressed(args.data, **result)
    print(f'Wrote {args.output} and {args.data}')


if __name__ == '__main__':
    main()
