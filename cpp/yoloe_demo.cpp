// YOLOE (Real-Time Seeing Anything, https://github.com/THU-MIG/yoloe)
// Open-vocabulary detector. Export with custom classes via ultralytics:
//   from ultralytics import YOLOE
//   model = YOLOE("yoloe-v8s-seg.pt")
//   names = ["person", "bus"]
//   model.set_classes(names, model.get_text_pe(names))
//   model.export(format="mnn")
// After export, output shape is [4 + N, 8400]. Pass the same names string to
// this binary so the labels match.
#include <stdio.h>
#include <string>
#include <vector>
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>

#include <cv/cv.hpp>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::CV;

static std::vector<std::string> split_names(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./yoloe_demo model.mnn input.jpg \"person,bus,...\" [forwardType] [precision] [thread]\n");
        return 0;
    }
    auto names = split_names(argv[3]);
    int thread = 4;
    int precision = 0;
    int forwardType = MNN_FORWARD_CPU;
    if (argc >= 5) forwardType = atoi(argv[4]);
    if (argc >= 6) precision = atoi(argv[5]);
    if (argc >= 7) thread = atoi(argv[6]);

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

    std::shared_ptr<Module> net(Module::load(std::vector<std::string>{}, std::vector<std::string>{}, argv[1], rtmgr));
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
    auto output = _Convert(outputs[0], NCHW);
    output = _Squeeze(output);
    // output shape: [4 + N, 8400]
    auto cx = _Gather(output, _Scalar<int>(0));
    auto cy = _Gather(output, _Scalar<int>(1));
    auto w = _Gather(output, _Scalar<int>(2));
    auto h = _Gather(output, _Scalar<int>(3));
    std::vector<int> startvals { 4, 0 };
    auto start = _Const(static_cast<void*>(startvals.data()), {2}, NCHW, halide_type_of<int>());
    std::vector<int> sizevals { -1, -1 };
    auto size = _Const(static_cast<void*>(sizevals.data()), {2}, NCHW, halide_type_of<int>());
    auto probs = _Slice(output, start, size);
    auto x0 = cx - w * _Const(0.5);
    auto y0 = cy - h * _Const(0.5);
    auto x1 = cx + w * _Const(0.5);
    auto y1 = cy + h * _Const(0.5);
    auto boxes = _Stack({x0, y0, x1, y1}, 1);
    auto scores = _ReduceMax(probs, {0});
    auto ids = _ArgMax(probs, 0);
    auto result_ids = _Nms(boxes, scores, 100, 0.45, 0.25);
    auto result_ptr = result_ids->readMap<int>();
    auto box_ptr = boxes->readMap<float>();
    auto ids_ptr = ids->readMap<int>();
    auto score_ptr = scores->readMap<float>();
    for (int i = 0; i < 100; i++) {
        auto idx = result_ptr[i];
        if (idx < 0) break;
        auto x0 = box_ptr[idx * 4 + 0] * scale;
        auto y0 = box_ptr[idx * 4 + 1] * scale;
        auto x1 = box_ptr[idx * 4 + 2] * scale;
        auto y1 = box_ptr[idx * 4 + 3] * scale;
        int class_idx = ids_ptr[idx];
        auto score = score_ptr[idx];
        const char* cls_name = (class_idx < (int)names.size()) ? names[class_idx].c_str() : "?";
        char label[128];
        snprintf(label, sizeof(label), "%s %.2f", cls_name, score);
        printf("### box: {%.1f, %.1f, %.1f, %.1f}, class: %s\n", x0, y0, x1, y1, label);
        rectangle(original_image, {x0, y0}, {x1, y1}, {0, 0, 255}, 2);
    }
    if (imwrite("res.jpg", original_image)) {
        MNN_PRINT("result image write to `res.jpg`.\n");
    }
    rtmgr->updateCache();
    return 0;
}
