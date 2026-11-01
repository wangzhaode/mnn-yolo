# Regression test

End-to-end check that every Python and C++ demo can load its expected MNN
model, run inference on `resource/test.jpg`, and write `res.jpg`.

## Run

```bash
./ci/test.sh                # full: prepare models + cpp build + all demos
./ci/test.sh --no-prepare   # reuse existing ci/models/
./ci/test.sh --python-only  # skip cpp
./ci/test.sh --cpp-only     # skip python
```

Exit code is `0` only if zero demos failed. Demos whose model could not be
prepared are recorded as `SKIP` and do not fail the run.

## What it does

1. `prepare_models.py`
   - legacy models such as `yolov5n / yolov6n / yolov7tiny / yolov9c` can be
     downloaded separately from [ModelScope](https://modelscope.cn/models/zhaode/mnn-yolo)
     and placed in `ci/models/` (these demos decode legacy ONNX shapes that
     ultralytics doesn't reproduce);
   - exports `yolov8n / yolov9t / yolov10n / yolo11n / yolo12n /
     yolo11n-seg / yolo11n-pose / yolo11n-obb` to ONNX and converts them
     with `MNNConvert`;
   - exports `yoloe-v8s-seg` with `set_classes(['person', 'bus'])` using the
     same ONNX-to-MNN route.
2. `cmake --build` for `cpp/build/`.
3. Runs each Python demo (`python/<demo>.py`) and each C++ demo
   (`cpp/build/<demo>`) once, captures stdout+stderr to `ci/logs/<label>.log`,
   and verifies `res.jpg` was written.

## Layout

```
ci/
  prepare_models.py    # builds the model set
  test.sh              # orchestrator
  models/              # *.mnn (gitignored)
  logs/                # per-demo logs + each demo's res.jpg
```

`yolov13` and other unsupported-by-ultralytics weights are not handled here
— run those demos manually with your own export.
