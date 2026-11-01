#!/usr/bin/env python3
"""
Prepare every .mnn model used by the regression test.

Pipeline
--------
1. ultralytics: `<weight>.pt` -> `<weight>.onnx`
2. MNNConvert CLI: `<weight>.onnx` -> `<weight>.mnn`

Why two steps instead of `model.export(format='mnn')`?
  Ultralytics' native MNN exporter calls `MNN.tools.mnnconvert.convert()`
  which only exists in some MNN-python versions. Going through ONNX +
  the standalone `MNNConvert` binary works against any reasonably modern
  MNN install.

What's NOT prepared here
------------------------
- v5 / v6 / v7 demos decode legacy ONNX shapes that ultralytics doesn't
  reproduce (yolov5 [25200, 85], yolov6 builtin-NMS, yolov7 [N, 7]). To
  regression-test those demos, supply your own .mnn manually.
- yolo13: not in ultralytics. Same — bring your own.

All artifacts go into ci/models/. .pt / .onnx by-products are removed so
only .mnn stays.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def _cleanup(stem: str) -> None:
    for ext in (".pt", ".onnx", ".torchscript"):
        p = MODELS / f"{stem}{ext}"
        if p.exists():
            p.unlink()
    for d in (MODELS / f"{stem}_ncnn_model", MODELS / f"{stem}_paddle_model"):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


def _onnx_to_mnn(onnx_path: Path, mnn_path: Path) -> bool:
    # --targetVersion 2.9 keeps the .mnn loadable by the older libMNN.dylib
    # vendored under cpp/libs/ (2.9.x). The Python MNN runtime is happy with
    # this format too, so a single conversion serves both paths.
    cmd = [
        "MNNConvert",
        "-f", "ONNX",
        "--modelFile", str(onnx_path),
        "--MNNModel", str(mnn_path),
        "--bizCode", "mnn-yolo-ci",
        "--targetVersion", "2.9",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        print("[fail]     MNNConvert binary not found in PATH")
        return False
    if r.returncode != 0 or not mnn_path.exists():
        print(f"[fail]     MNNConvert: {r.stderr.strip().splitlines()[-1] if r.stderr else r.stdout.strip()}")
        return False
    return True


def _export(weight: str, *, set_classes: list[str] | None = None) -> bool:
    stem = Path(weight).stem
    target = MODELS / f"{stem}.mnn"
    if target.exists() and target.stat().st_size > 0:
        print(f"[skip]     {target.name} already exists")
        return True

    cwd = Path.cwd()
    os.chdir(MODELS)
    try:
        if set_classes is not None:
            from ultralytics import YOLOE
            model = YOLOE(weight)
            model.set_classes(set_classes, model.get_text_pe(set_classes))
        else:
            from ultralytics import YOLO
            model = YOLO(weight)
        onnx_out = Path(model.export(format="onnx", opset=12))
        if not onnx_out.is_absolute():
            onnx_out = MODELS / onnx_out.name
        if not _onnx_to_mnn(onnx_out, target):
            return False
        print(f"[ok]       {target.name}")
        return True
    except Exception as e:
        print(f"[fail]     {stem}: {e}")
        return False
    finally:
        os.chdir(cwd)
        _cleanup(stem)


def main() -> int:
    successes: list[str] = []
    failures: list[str] = []

    for weight, classes in [
        ("yolov8n.pt",       None),
        ("yolov9t.pt",       None),
        ("yolov10n.pt",      None),
        ("yolo11n.pt",       None),
        ("yolo12n.pt",       None),
        ("yolo11n-seg.pt",   None),
        ("yolo11n-pose.pt",  None),
        ("yolo11n-obb.pt",   None),
        ("yoloe-v8s-seg.pt", ["person", "bus"]),
    ]:
        ok = _export(weight, set_classes=classes)
        (successes if ok else failures).append(Path(weight).stem)

    (MODELS / "yoloe_names.txt").write_text("person,bus\n")

    print("\n=== prepare_models summary ===")
    print(f"ok    ({len(successes)}): {', '.join(successes) or '-'}")
    print(f"fail  ({len(failures)}): {', '.join(failures) or '-'}")
    print(f"models dir: {MODELS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
