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

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./yolov6_demo.out model.mnn input.jpg [forwardType] [precision] [thread]\n");
        return 0;
    }
    int thread = 4;
    int precision = 0;
    int forwardType = MNN_FORWARD_CPU;
    if (argc >= 4) {
        forwardType = atoi(argv[3]);
    }
    if (argc >= 5) {
        precision = atoi(argv[4]);
    }
    if (argc >= 6) {
        thread = atoi(argv[5]);
    }
    MNN::ScheduleConfig sConfig;
    sConfig.type = static_cast<MNNForwardType>(forwardType);
    sConfig.numThread = thread;
    BackendConfig bConfig;
    bConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
    sConfig.backendConfig = &bConfig;
    std::shared_ptr<Executor::RuntimeManager> rtmgr = std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(sConfig));
    if(rtmgr == nullptr) {
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
    auto det_boxes = _Convert(outputs[0], NCHW);
    auto det_classes = _Convert(outputs[1], NCHW);
    auto det_scores = _Convert(outputs[2], NCHW);
    auto num_dets = _Convert(outputs[3], NCHW);
    auto boxes_ptr = det_boxes->readMap<float>();
    auto class_ptr = det_classes->readMap<int>();
    auto score_ptr = det_scores->readMap<float>();
    int box_num = num_dets->readMap<int>()[0];
    for (int i = 0; i < box_num; i++) {
        auto box_ptr = boxes_ptr + i * 4;
        auto x0 = box_ptr[0] * scale;
        auto y0 = box_ptr[1] * scale;
        auto x1 = box_ptr[2] * scale;
        auto y1 = box_ptr[3] * scale;
        auto class_idx = class_ptr[i];
        auto score = score_ptr[i];
        printf("### box: {%f, %f, %f, %f}, class_idx: %d, score: %f\n", x0, y0, x1, y1, class_idx, score);
        rectangle(original_image, {x0, y0}, {x1, y1}, {0, 0, 255}, 2);
    }
    if (imwrite("res.jpg", original_image)) {
        MNN_PRINT("result image write to `res.jpg`.\n");
    }
    rtmgr->updateCache();
    return 0;
}
