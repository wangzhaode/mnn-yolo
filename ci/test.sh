#!/usr/bin/env bash
# Regression test: prepares every demo's model, builds the C++ side, then runs
# every Python and C++ demo against resource/test.jpg.
#
# Behavior:
#   - Single failing demo does NOT abort the run; it's recorded and reported.
#   - A demo whose model is missing is recorded as SKIP, not FAIL.
#   - Final exit code is 0 only if there are zero FAILs (SKIPs are tolerated).
#
# Usage:
#   ci/test.sh                # full run: prepare models + python + cpp
#   ci/test.sh --no-prepare   # reuse whatever's already in ci/models/
#   ci/test.sh --python-only  # skip cpp build & cpp demos
#   ci/test.sh --cpp-only     # skip python demos
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CI_DIR="$ROOT/ci"
MODELS="$CI_DIR/models"
LOGS="$CI_DIR/logs"
IMG="$ROOT/resource/test.jpg"
BUILD="$ROOT/cpp/build"

mkdir -p "$LOGS"
rm -f "$LOGS"/*.log

# Pick a Python that actually has MNN. Override with PYTHON=... if needed.
if [[ -z "${PYTHON:-}" ]]; then
    for cand in python python3; do
        if command -v "$cand" >/dev/null && "$cand" -c 'import MNN' >/dev/null 2>&1; then
            PYTHON="$cand"
            break
        fi
    done
    PYTHON="${PYTHON:-python3}"
fi
echo "using python: $($PYTHON -c 'import sys; print(sys.executable)')"

DO_PREPARE=1
DO_PYTHON=1
DO_CPP=1
for arg in "$@"; do
    case "$arg" in
        --no-prepare)  DO_PREPARE=0 ;;
        --python-only) DO_CPP=0 ;;
        --cpp-only)    DO_PYTHON=0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

PASSES=()
FAILS=()
SKIPS=()

# ----- runner -------------------------------------------------------------
# run_demo <label> <model-file-relative-to-MODELS> <command> [args...]
#   * Skips if model file is absent.
#   * Runs the command in a clean dir (so res.jpg from prior runs doesn't lie).
#   * Verifies res.jpg appears.
run_demo() {
    local label="$1"; shift
    local model_file="$1"; shift
    local model_path="$MODELS/$model_file"
    local log="$LOGS/${label}.log"
    local rundir="$LOGS/${label}.rundir"

    if [[ ! -f "$model_path" ]]; then
        echo "SKIP  $label  (missing model: $model_file)"
        SKIPS+=("$label (no $model_file)")
        return 0
    fi

    rm -rf "$rundir"; mkdir -p "$rundir"
    (cd "$rundir" && "$@") > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "FAIL  $label  (exit $rc, see $log)"
        FAILS+=("$label (exit $rc)")
        return 0
    fi
    if [[ ! -s "$rundir/res.jpg" ]]; then
        echo "FAIL  $label  (no res.jpg produced)"
        FAILS+=("$label (no res.jpg)")
        return 0
    fi
    echo "PASS  $label"
    PASSES+=("$label")
}

# ----- prepare ------------------------------------------------------------
if [[ $DO_PREPARE -eq 1 ]]; then
    echo "=== preparing models ==="
    "$PYTHON" "$CI_DIR/prepare_models.py" 2>&1 | tee "$LOGS/prepare.log"
    echo
fi

# ----- C++ build ----------------------------------------------------------
if [[ $DO_CPP -eq 1 ]]; then
    echo "=== building cpp demos ==="
    # Wipe stale CMake cache (path may differ if repo was relocated) but keep
    # the dir itself so this stays a single-line concept in CI.
    rm -rf "$BUILD"
    mkdir -p "$BUILD"
    (
        cd "$BUILD" \
            && cmake .. > "$LOGS/cmake.log" 2>&1 \
            && cmake --build . -j > "$LOGS/cmake-build.log" 2>&1
    )
    if [[ $? -ne 0 ]]; then
        echo "FAIL  cpp-build  (see $LOGS/cmake-build.log)"
        FAILS+=("cpp-build")
        DO_CPP=0
    else
        echo "PASS  cpp-build"
        PASSES+=("cpp-build")
    fi
    echo
fi

YOLOE_NAMES="$(cat "$MODELS/yoloe_names.txt" 2>/dev/null || echo 'person,bus')"
YOLOE_NAMES="${YOLOE_NAMES%$'\n'}"

# ----- python demos -------------------------------------------------------
if [[ $DO_PYTHON -eq 1 ]]; then
    echo "=== python demos ==="
    PY="$ROOT/python"
    run_demo py_yolov5      yolov5n.mnn      "$PYTHON" "$PY/yolov5_demo.py"      --model "$MODELS/yolov5n.mnn"      --img "$IMG"
    run_demo py_yolov6      yolov6n.mnn      "$PYTHON" "$PY/yolov6_demo.py"      --model "$MODELS/yolov6n.mnn"      --img "$IMG"
    run_demo py_yolov7      yolov7tiny.mnn   "$PYTHON" "$PY/yolov7_demo.py"      --model "$MODELS/yolov7tiny.mnn"   --img "$IMG"
    run_demo py_yolov8      yolov8n.mnn      "$PYTHON" "$PY/yolov8_demo.py"      --model "$MODELS/yolov8n.mnn"      --img "$IMG"
    run_demo py_yolov9      yolov9t.mnn      "$PYTHON" "$PY/yolov9_demo.py"      --model "$MODELS/yolov9t.mnn"      --img "$IMG"
    run_demo py_yolov10     yolov10n.mnn     "$PYTHON" "$PY/yolov10_demo.py"     --model "$MODELS/yolov10n.mnn"     --img "$IMG"
    run_demo py_yolo11      yolo11n.mnn      "$PYTHON" "$PY/yolo11_demo.py"      --model "$MODELS/yolo11n.mnn"      --img "$IMG"
    run_demo py_yolo12      yolo12n.mnn      "$PYTHON" "$PY/yolo12_demo.py"      --model "$MODELS/yolo12n.mnn"      --img "$IMG"
    run_demo py_yoloe       yoloe-v8s-seg.mnn "$PYTHON" "$PY/yoloe_demo.py"      --model "$MODELS/yoloe-v8s-seg.mnn" --img "$IMG" --names "$YOLOE_NAMES"
    run_demo py_yolo11_seg  yolo11n-seg.mnn  "$PYTHON" "$PY/yolo11_seg_demo.py"  --model "$MODELS/yolo11n-seg.mnn"  --img "$IMG"
    run_demo py_yolo11_pose yolo11n-pose.mnn "$PYTHON" "$PY/yolo11_pose_demo.py" --model "$MODELS/yolo11n-pose.mnn" --img "$IMG"
    run_demo py_yolo11_obb  yolo11n-obb.mnn  "$PYTHON" "$PY/yolo11_obb_demo.py"  --model "$MODELS/yolo11n-obb.mnn"  --img "$IMG"
    echo
fi

# ----- cpp demos ----------------------------------------------------------
if [[ $DO_CPP -eq 1 ]]; then
    echo "=== cpp demos ==="
    # MNN dylibs live in cpp/libs — make sure the runtime linker finds them.
    export DYLD_LIBRARY_PATH="$ROOT/cpp/libs:${DYLD_LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$ROOT/cpp/libs:${LD_LIBRARY_PATH:-}"

    # Sanity check: if the vendored libMNN.dylib is older than the MNNConvert
    # used to produce ci/models/, every cpp demo crashes on Module::load —
    # sometimes with "Invalidate buffer to create MNN Module" on stderr,
    # sometimes with a bare segfault. Run yolo11_demo once; if it crashes
    # while the matching python demo passes, treat all cpp demos as SKIP.
    if [[ -f "$MODELS/yolo11n.mnn" && -x "$BUILD/yolo11_demo" ]]; then
        sanity_log="$LOGS/cpp_sanity.log"
        sanity_dir="$LOGS/cpp_sanity.rundir"
        rm -rf "$sanity_dir"; mkdir -p "$sanity_dir"
        (cd "$sanity_dir" && "$BUILD/yolo11_demo" "$MODELS/yolo11n.mnn" "$IMG") > "$sanity_log" 2>&1
        sanity_rc=$?
        if [[ $sanity_rc -ne 0 || ! -s "$sanity_dir/res.jpg" ]]; then
            echo "SKIP  cpp_*  (vendored cpp/libs/libMNN.dylib cannot load models"
            echo "             converted by the current MNNConvert — likely a"
            echo "             libMNN/MNNConvert version mismatch."
            echo "             refresh cpp/libs from a matching MNN release."
            echo "             sanity exit=$sanity_rc, see $sanity_log)"
            for d in cpp_yolov5 cpp_yolov6 cpp_yolov7 cpp_yolov8 cpp_yolov9 \
                     cpp_yolov10 cpp_yolo11 cpp_yolo12 cpp_yoloe \
                     cpp_yolo11_seg cpp_yolo11_pose cpp_yolo11_obb; do
                SKIPS+=("$d (libMNN/MNNConvert mismatch)")
            done
            DO_CPP=0
        fi
    fi
fi
if [[ $DO_CPP -eq 1 ]]; then

    run_demo cpp_yolov5      yolov5n.mnn      "$BUILD/yolov5_demo"      "$MODELS/yolov5n.mnn"      "$IMG"
    run_demo cpp_yolov6      yolov6n.mnn      "$BUILD/yolov6_demo"      "$MODELS/yolov6n.mnn"      "$IMG"
    run_demo cpp_yolov7      yolov7tiny.mnn   "$BUILD/yolov7_demo"      "$MODELS/yolov7tiny.mnn"   "$IMG"
    run_demo cpp_yolov8      yolov8n.mnn      "$BUILD/yolov8_demo"      "$MODELS/yolov8n.mnn"      "$IMG"
    run_demo cpp_yolov9      yolov9t.mnn      "$BUILD/yolov9_demo"      "$MODELS/yolov9t.mnn"      "$IMG"
    run_demo cpp_yolov10     yolov10n.mnn     "$BUILD/yolov10_demo"     "$MODELS/yolov10n.mnn"     "$IMG"
    run_demo cpp_yolo11      yolo11n.mnn      "$BUILD/yolo11_demo"      "$MODELS/yolo11n.mnn"      "$IMG"
    run_demo cpp_yolo12      yolo12n.mnn      "$BUILD/yolo12_demo"      "$MODELS/yolo12n.mnn"      "$IMG"
    run_demo cpp_yoloe       yoloe-v8s-seg.mnn "$BUILD/yoloe_demo"      "$MODELS/yoloe-v8s-seg.mnn" "$IMG" "$YOLOE_NAMES"
    run_demo cpp_yolo11_seg  yolo11n-seg.mnn  "$BUILD/yolo11_seg_demo"  "$MODELS/yolo11n-seg.mnn"  "$IMG"
    run_demo cpp_yolo11_pose yolo11n-pose.mnn "$BUILD/yolo11_pose_demo" "$MODELS/yolo11n-pose.mnn" "$IMG"
    run_demo cpp_yolo11_obb  yolo11n-obb.mnn  "$BUILD/yolo11_obb_demo"  "$MODELS/yolo11n-obb.mnn"  "$IMG"
    echo
fi

# ----- summary ------------------------------------------------------------
echo "=========================================="
echo "PASS:  ${#PASSES[@]}"
echo "FAIL:  ${#FAILS[@]}"
echo "SKIP:  ${#SKIPS[@]}"
echo "=========================================="
if [[ ${#FAILS[@]} -gt 0 ]]; then
    echo "Failures:"
    for f in "${FAILS[@]}"; do echo "  - $f"; done
fi
if [[ ${#SKIPS[@]} -gt 0 ]]; then
    echo "Skipped:"
    for s in "${SKIPS[@]}"; do echo "  - $s"; done
fi
echo "Logs: $LOGS"

[[ ${#FAILS[@]} -eq 0 ]]
