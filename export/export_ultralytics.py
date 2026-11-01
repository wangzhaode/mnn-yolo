#!/usr/bin/env python3
"""Export an Ultralytics checkpoint to ONNX, then convert it with MNNConvert."""
import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights', help='local .pt checkpoint or Ultralytics model name')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640], help='one square size or height width')
    parser.add_argument('--opset', type=int, default=12)
    parser.add_argument('--converter', default='MNNConvert')
    parser.add_argument('--output', help='destination .mnn file (default: next to ONNX)')
    args = parser.parse_args()
    if len(args.imgsz) not in (1, 2) or min(args.imgsz) <= 0:
        parser.error('--imgsz must contain one or two positive integers')
    if not shutil.which(args.converter) and not Path(args.converter).is_file():
        parser.error(f'MNN converter not found: {args.converter}')
    from ultralytics import YOLO
    model = YOLO(args.weights)
    onnx = Path(model.export(format='onnx', imgsz=args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz,
                             opset=args.opset, dynamic=False, simplify=False, nms=False))
    output = Path(args.output) if args.output else onnx.with_suffix('.mnn')
    if output.resolve() == onnx.resolve():
        parser.error('output must differ from ONNX file')
    subprocess.run([args.converter, '-f', 'ONNX', '--modelFile', str(onnx), '--MNNModel', str(output),
                    '--bizCode', 'mnn-yolo'], check=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f'MNNConvert did not produce {output}')
    print(f'Converted {onnx} -> {output}')


if __name__ == '__main__':
    main()
