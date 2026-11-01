// YOLO11-obb (Oriented Bounding Box, DOTA-v1, 15 classes)
// Output: [1, 20, anchors] -> 4 box (cx, cy, w, h) + 15 class probs + 1 angle (rad)
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

static const char* DOTA_NAMES[] = {
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool",
};

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./yolo11_obb_demo model.mnn input.jpg [forwardType] [precision] [thread]\n");
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

    std::shared_ptr<Module> net(Module::load(std::vector<std::string>{}, std::vector<std::string>{}, argv[1], rtmgr));
    auto original_image = imread(argv[2]);
    auto dims = original_image->getInfo()->dim;
    int ih = dims[0];
    int iw = dims[1];
    int len = ih > iw ? ih : iw;
    const int IMGSZ = 1024;
    float scale = len / static_cast<float>(IMGSZ);
    std::vector<int> padvals { 0, len - ih, 0, len - iw, 0, 0 };
    auto pads = _Const(static_cast<void*>(padvals.data()), {3, 2}, NCHW, halide_type_of<int>());
    auto image = _Pad(original_image, pads, CONSTANT);
    image = resize(image, Size(IMGSZ, IMGSZ), 0, 0, INTER_LINEAR, -1, {0., 0., 0.}, {1./255., 1./255., 1./255.});
    auto input = _Unsqueeze(image, {0});
    input = _Convert(input, NC4HW4);
    auto outputs = net->onForward({input});
    auto output = _Squeeze(_Convert(outputs[0], NCHW));   // [20, anchors]

    auto cx = _Gather(output, _Scalar<int>(0));
    auto cy = _Gather(output, _Scalar<int>(1));
    auto w  = _Gather(output, _Scalar<int>(2));
    auto h  = _Gather(output, _Scalar<int>(3));
    std::vector<int> clsStart { 4, 0 };
    std::vector<int> clsSize  { 15, -1 };
    auto cls_start = _Const(static_cast<void*>(clsStart.data()), {2}, NCHW, halide_type_of<int>());
    auto cls_size  = _Const(static_cast<void*>(clsSize.data()),  {2}, NCHW, halide_type_of<int>());
    auto cls_probs = _Slice(output, cls_start, cls_size);
    auto angle = _Gather(output, _Scalar<int>(19));

    auto x0 = cx - w * _Const(0.5);
    auto y0 = cy - h * _Const(0.5);
    auto x1 = cx + w * _Const(0.5);
    auto y1 = cy + h * _Const(0.5);
    auto aabb = _Stack({x0, y0, x1, y1}, 1);
    auto scores = _ReduceMax(cls_probs, {0});
    auto ids = _ArgMax(cls_probs, 0);
    auto result_ids = _Nms(aabb, scores, 100, 0.45, 0.25);

    auto result_ptr = result_ids->readMap<int>();
    auto cx_ptr = cx->readMap<float>();
    auto cy_ptr = cy->readMap<float>();
    auto w_ptr  = w->readMap<float>();
    auto h_ptr  = h->readMap<float>();
    auto angle_ptr = angle->readMap<float>();
    auto score_ptr = scores->readMap<float>();
    auto ids_ptr   = ids->readMap<int>();

    for (int i = 0; i < 100; i++) {
        auto idx = result_ptr[i];
        if (idx < 0) break;
        float rcx = cx_ptr[idx] * scale;
        float rcy = cy_ptr[idx] * scale;
        float rw  = w_ptr[idx] * scale;
        float rh  = h_ptr[idx] * scale;
        float ang = angle_ptr[idx];
        int cls_idx = ids_ptr[idx];
        float score = score_ptr[idx];
        float cos_a = cosf(ang), sin_a = sinf(ang);
        float hw = rw * 0.5f, hh = rh * 0.5f;
        float dxs[4] = {-hw,  hw, hw, -hw};
        float dys[4] = {-hh, -hh, hh,  hh};
        Point corners[4];
        for (int k = 0; k < 4; ++k) {
            corners[k] = {rcx + dxs[k] * cos_a - dys[k] * sin_a,
                          rcy + dxs[k] * sin_a + dys[k] * cos_a};
        }
        for (int k = 0; k < 4; ++k) {
            line(original_image, corners[k], corners[(k + 1) % 4], {0, 0, 255}, 2);
        }
        char label[128];
        snprintf(label, sizeof(label), "%s %.2f", DOTA_NAMES[cls_idx], score);
        printf("### obb: cx=%.1f cy=%.1f w=%.1f h=%.1f angle=%.3frad, class: %s\n",
               rcx, rcy, rw, rh, ang, label);
    }
    if (imwrite("res.jpg", original_image)) {
        MNN_PRINT("result image write to `res.jpg`.\n");
    }
    rtmgr->updateCache();
    return 0;
}
