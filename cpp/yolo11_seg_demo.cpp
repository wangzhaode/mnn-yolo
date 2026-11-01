// YOLO11-seg (instance segmentation)
// Two outputs:
//   output0: [1, 116, 8400] = 4 box + 80 class probs + 32 mask coefficients
//   output1: [1, 32, 160, 160] = mask prototypes
// Per-instance mask = sigmoid(coef @ proto.reshape(32, -1)). For brevity this
// demo prints mask_mean per detection and draws box + label only; extend with
// resize/threshold/overlay as needed.
#include <stdio.h>
#include <math.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>

#include <cv/cv.hpp>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::CV;

static const char* COCO_NAMES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./yolo11_seg_demo model.mnn input.jpg [forwardType] [precision] [thread]\n");
        return 0;
    }
    int thread = 4;
    int precision = 0;
    int forwardType = MNN_FORWARD_CPU;
    if (argc >= 4) forwardType = atoi(argv[3]);
    if (argc >= 5) precision = atoi(argv[4]);
    if (argc >= 6) thread = atoi(argv[5]);

    MNN::ScheduleConfig sConfig;
    sConfig.type = static_cast<MNNForwardType>(forwardType);
    sConfig.numThread = thread;
    BackendConfig bConfig;
    bConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
    sConfig.backendConfig = &bConfig;
    std::shared_ptr<Executor::RuntimeManager> rtmgr(Executor::RuntimeManager::createRuntimeManager(sConfig));
    if (rtmgr == nullptr) {
        MNN_ERROR("Empty RuntimeManger\n");
        return 0;
    }
    rtmgr->setCache(".cachefile");

    // Multi-output model: explicit input/output names are required so
    // onForward returns both `det` and `proto`.
    std::shared_ptr<Module> net(Module::load({"images"}, {"output0", "output1"}, argv[1], rtmgr));
    auto original_image = imread(argv[2]);
    auto dims = original_image->getInfo()->dim;
    int ih = dims[0];
    int iw = dims[1];
    int len = ih > iw ? ih : iw;
    float scale = len / 640.0;
    std::vector<int> padvals { 0, len - ih, 0, len - iw, 0, 0 };
    auto pads = _Const(static_cast<void*>(padvals.data()), {3, 2}, NCHW, halide_type_of<int>());
    auto image = _Pad(original_image, pads, CONSTANT);
    image = resize(image, Size(640, 640), 0, 0, INTER_LINEAR, -1, {0., 0., 0.}, {1./255., 1./255., 1./255.});
    auto input = _Unsqueeze(image, {0});
    input = _Convert(input, NC4HW4);
    auto outputs = net->onForward({input});
    auto det = _Squeeze(_Convert(outputs[0], NCHW));     // [116, 8400]
    auto proto = _Squeeze(_Convert(outputs[1], NCHW));   // [32, 160, 160]

    auto cx = _Gather(det, _Scalar<int>(0));
    auto cy = _Gather(det, _Scalar<int>(1));
    auto w  = _Gather(det, _Scalar<int>(2));
    auto h  = _Gather(det, _Scalar<int>(3));
    // class probs: rows 4..83
    std::vector<int> clsStart { 4, 0 };
    std::vector<int> clsSize  { 80, -1 };
    auto cls_start = _Const(static_cast<void*>(clsStart.data()), {2}, NCHW, halide_type_of<int>());
    auto cls_size  = _Const(static_cast<void*>(clsSize.data()),  {2}, NCHW, halide_type_of<int>());
    auto cls_probs = _Slice(det, cls_start, cls_size);
    // mask coefs: rows 84..115 -> [32, 8400]
    std::vector<int> mcStart { 84, 0 };
    std::vector<int> mcSize  { 32, -1 };
    auto mc_start = _Const(static_cast<void*>(mcStart.data()), {2}, NCHW, halide_type_of<int>());
    auto mc_size  = _Const(static_cast<void*>(mcSize.data()),  {2}, NCHW, halide_type_of<int>());
    auto mask_coefs = _Slice(det, mc_start, mc_size);

    auto x0 = cx - w * _Const(0.5);
    auto y0 = cy - h * _Const(0.5);
    auto x1 = cx + w * _Const(0.5);
    auto y1 = cy + h * _Const(0.5);
    auto boxes = _Stack({x0, y0, x1, y1}, 1);
    auto scores = _ReduceMax(cls_probs, {0});
    auto ids = _ArgMax(cls_probs, 0);
    auto result_ids = _Nms(boxes, scores, 100, 0.45, 0.25);

    auto result_ptr = result_ids->readMap<int>();
    auto box_ptr = boxes->readMap<float>();
    auto ids_ptr = ids->readMap<int>();
    auto score_ptr = scores->readMap<float>();
    auto coef_ptr = mask_coefs->readMap<float>();   // [32, 8400] row-major
    auto proto_ptr = proto->readMap<float>();       // [32, 160, 160]
    const int proto_hw = 160 * 160;

    for (int i = 0; i < 100; i++) {
        auto idx = result_ptr[i];
        if (idx < 0) break;
        auto x0 = box_ptr[idx * 4 + 0] * scale;
        auto y0 = box_ptr[idx * 4 + 1] * scale;
        auto x1 = box_ptr[idx * 4 + 2] * scale;
        auto y1 = box_ptr[idx * 4 + 3] * scale;
        int class_idx = ids_ptr[idx];
        float score = score_ptr[idx];

        // mean activation of the mask logits as a sanity check
        double sum = 0.0;
        for (int p = 0; p < proto_hw; ++p) {
            float v = 0.f;
            for (int c = 0; c < 32; ++c) {
                v += coef_ptr[c * 8400 + idx] * proto_ptr[c * proto_hw + p];
            }
            sum += v;
        }
        float mean_act = static_cast<float>(sum / proto_hw);

        char label[128];
        snprintf(label, sizeof(label), "%s %.2f", COCO_NAMES[class_idx], score);
        printf("### box: {%.1f, %.1f, %.1f, %.1f}, class: %s, mask_mean: %.3f\n",
               x0, y0, x1, y1, label, mean_act);
        rectangle(original_image, {x0, y0}, {x1, y1}, {0, 0, 255}, 2);
    }
    if (imwrite("res.jpg", original_image)) {
        MNN_PRINT("result image write to `res.jpg`.\n");
    }
    rtmgr->updateCache();
    return 0;
}
