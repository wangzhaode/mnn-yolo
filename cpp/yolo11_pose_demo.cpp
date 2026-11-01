// YOLO11-pose (human pose, 17 COCO keypoints)
// Output: [1, 56, 8400] -> 4 box + 1 person conf + 17 * 3 (kx, ky, conf)
#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>

#include <cv/cv.hpp>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::CV;

static const int SKELETON[][2] = {
    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
    {5, 11},  {6, 12},  {5, 6},   {5, 7},   {6, 8},
    {7, 9},   {8, 10},  {1, 2},   {0, 1},   {0, 2},
    {1, 3},   {2, 4},   {3, 5},   {4, 6},
};
static const int SKELETON_LEN = sizeof(SKELETON) / sizeof(SKELETON[0]);

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./yolo11_pose_demo model.mnn input.jpg [forwardType] [precision] [thread]\n");
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
    float scale = len / 640.0;
    std::vector<int> padvals { 0, len - ih, 0, len - iw, 0, 0 };
    auto pads = _Const(static_cast<void*>(padvals.data()), {3, 2}, NCHW, halide_type_of<int>());
    auto image = _Pad(original_image, pads, CONSTANT);
    image = resize(image, Size(640, 640), 0, 0, INTER_LINEAR, -1, {0., 0., 0.}, {1./255., 1./255., 1./255.});
    auto input = _Unsqueeze(image, {0});
    input = _Convert(input, NC4HW4);
    auto outputs = net->onForward({input});
    auto output = _Squeeze(_Convert(outputs[0], NCHW));   // [56, 8400]

    auto cx = _Gather(output, _Scalar<int>(0));
    auto cy = _Gather(output, _Scalar<int>(1));
    auto w  = _Gather(output, _Scalar<int>(2));
    auto h  = _Gather(output, _Scalar<int>(3));
    auto person_score = _Gather(output, _Scalar<int>(4));
    // keypoints rows 5..55 -> [51, 8400]
    std::vector<int> kpStart { 5, 0 };
    std::vector<int> kpSize  { 51, -1 };
    auto kp_start = _Const(static_cast<void*>(kpStart.data()), {2}, NCHW, halide_type_of<int>());
    auto kp_size  = _Const(static_cast<void*>(kpSize.data()),  {2}, NCHW, halide_type_of<int>());
    auto kpts = _Slice(output, kp_start, kp_size);

    auto x0 = cx - w * _Const(0.5);
    auto y0 = cy - h * _Const(0.5);
    auto x1 = cx + w * _Const(0.5);
    auto y1 = cy + h * _Const(0.5);
    auto boxes = _Stack({x0, y0, x1, y1}, 1);
    auto result_ids = _Nms(boxes, person_score, 100, 0.45, 0.25);

    auto result_ptr = result_ids->readMap<int>();
    auto box_ptr = boxes->readMap<float>();
    auto score_ptr = person_score->readMap<float>();
    auto kp_ptr = kpts->readMap<float>();   // row-major [51, 8400]

    for (int i = 0; i < 100; i++) {
        auto idx = result_ptr[i];
        if (idx < 0) break;
        auto bx0 = box_ptr[idx * 4 + 0] * scale;
        auto by0 = box_ptr[idx * 4 + 1] * scale;
        auto bx1 = box_ptr[idx * 4 + 2] * scale;
        auto by1 = box_ptr[idx * 4 + 3] * scale;
        float score = score_ptr[idx];
        // gather 17 keypoints
        struct { float x, y, c; } pts[17];
        for (int k = 0; k < 17; ++k) {
            pts[k].x = kp_ptr[(k * 3 + 0) * 8400 + idx] * scale;
            pts[k].y = kp_ptr[(k * 3 + 1) * 8400 + idx] * scale;
            pts[k].c = kp_ptr[(k * 3 + 2) * 8400 + idx];
            if (pts[k].c > 0.5) {
                circle(original_image, {pts[k].x, pts[k].y}, 3, {0, 255, 0}, -1);
            }
        }
        for (int e = 0; e < SKELETON_LEN; ++e) {
            int a = SKELETON[e][0], b = SKELETON[e][1];
            if (pts[a].c > 0.5 && pts[b].c > 0.5) {
                line(original_image, {pts[a].x, pts[a].y}, {pts[b].x, pts[b].y},
                     {255, 0, 0}, 2);
            }
        }
        char label[64];
        snprintf(label, sizeof(label), "person %.2f", score);
        printf("### box: {%.1f, %.1f, %.1f, %.1f}, %s\n", bx0, by0, bx1, by1, label);
        rectangle(original_image, {bx0, by0}, {bx1, by1}, {0, 0, 255}, 2);
    }
    if (imwrite("res.jpg", original_image)) {
        MNN_PRINT("result image write to `res.jpg`.\n");
    }
    rtmgr->updateCache();
    return 0;
}
