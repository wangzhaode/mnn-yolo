#-- coding:utf8 --
import argparse

import MNN
import MNN.numpy as np
import MNN.cv as cv2

def inference(model, img, precision, backend, thread):
    config = {}
    config['precision'] = precision
    config['backend'] = backend
    config['numThread'] = thread
    rt = MNN.nn.create_runtime_manager((config,))
    # net = MNN.nn.load_module_from_file(model, ['images'], ['output0'], runtime_manager=rt)
    net = MNN.nn.load_module_from_file(model, [], [], runtime_manager=rt)
    original_image = cv2.imread(img)
    ih, iw, _ = original_image.shape
    length = max((ih, iw))
    scale = length / 640
    image = np.pad(original_image, [[0, length - ih], [0, length - iw], [0, 0]], 'constant')
    image = cv2.resize(image, (640, 640), 0., 0., cv2.INTER_LINEAR, -1, [0., 0., 0.], [1./255., 1./255., 1./255.])
    input_var = np.expand_dims(image, 0)
    input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
    output_var = net.forward(input_var)
    output_var = MNN.expr.convert(output_var, MNN.expr.NCHW)
    # output shape: [box_num, 7]; 7 means: [batch_id, x0, y0, x1, y1, class_idx, score]
    box_num = output_var.shape[0]
    result_data = output_var.read_as_tuple()
    for i in range(box_num):
        batch_id = result_data[7 * i + 0]
        x0 = result_data[7 * i + 1]
        y0 = result_data[7 * i + 2]
        x1 = result_data[7 * i + 3]
        y1 = result_data[7 * i + 4]
        class_idx = result_data[7 * i + 5]
        score = result_data[7 * i + 6]
        y0 = int(y0 * scale)
        y1 = int(y1 * scale)
        x0 = int(x0 * scale)
        x1 = int(x1 * scale)
        print(f"### class_idx: {class_idx}, score: {score}")
        cv2.rectangle(original_image, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite('res.jpg', original_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='the mobilenet model path')
    parser.add_argument('--img', type=str, required=True, help='the input image path')
    parser.add_argument('--precision', type=str, default='normal', help='inference precision: normal, low, high, lowBF')
    parser.add_argument('--backend', type=str, default='CPU', help='inference backend: CPU, OPENCL, OPENGL, NN, VULKAN, METAL, TRT, CUDA, HIAI')
    parser.add_argument('--thread', type=int, default=4, help='inference using thread: int')
    args = parser.parse_args()
    inference(args.model, args.img, args.precision, args.backend, args.thread)
