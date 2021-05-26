#include "yolo_onnx.h"

#include <bits/c++config.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>

#include "bounding_box.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "opencv2/dnn/dnn.hpp"
#include "opencv2/videoio.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

void YoloOnnx::load_model(const std::string& filename) {
    if (env_ == nullptr) {
        throw logic_error("env cannot be null");
    }
    session_ =
        Ort::Session(*env_, filename.c_str(), Ort::SessionOptions{nullptr});
}

const char* YoloOnnx::get_input_name() {
    check_input_count();

    return session_.GetInputName(0, allocator_);
}

std::vector<int64_t> YoloOnnx::get_input_shape() {
    check_input_count();

    Ort::TypeInfo type_info = session_.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetShape();
}

std::vector<BoundingBox> YoloOnnx::detect(cv::InputArray image) {
    // [N, C, H, W]
    vector<int64_t> input_shape = get_input_shape();

    blobFromImage(image, blob_, 1.0 / 255.0,
                  {(int)input_shape.at(3), (int)input_shape.at(2)}, {}, true);
    int tensor_size = input_shape.at(1) * input_shape.at(2) * input_shape.at(3);

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float*)blob_.data, tensor_size, input_shape.data(),
        input_shape.size());

    const char* input_name = get_input_name();
    const char* output_name = output_name_.c_str();
    auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, &input_name,
                                       &input_tensor, 1, &output_name, 1);
    Ort::Value& output_tensor = output_tensors.at(0);
    vector<int64_t> output_size =
        output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    if (output_size.size() != 3) {
        throw logic_error("Expect output dim == 3");
    }
    int class_count = output_size.at(2) - 5;

    Mat output_mat(output_size[0], output_size[1], CV_32FC(output_size[2]),
                   output_tensor.GetTensorMutableData<float>());

    Size2f ratio((float)image.cols() / input_shape[3],
                 (float)image.rows() / input_shape[2]);
    Rect image_rect(0, 0, image.cols(), image.rows());

    vector<vector<Rect2d>> bboxes(class_count);
    vector<vector<float>> scores(class_count);

    for (int i = 0; i < output_mat.rows; ++i) {
        for (int j = 0; j < output_mat.cols; ++j) {
            float* ptr = output_mat.ptr<float>(i, j);
            float conf = ptr[4];

            if (conf > threshold_) {
                float cx = ptr[0] * ratio.width;
                float cy = ptr[1] * ratio.height;
                float w = ptr[2] * ratio.width;
                float h = ptr[3] * ratio.height;

                float x = cx - w / 2;
                float y = cy - h / 2;
                Rect2d rect(x, y, w, h);

                Mat classes(1, class_count, CV_32F, ptr + 5);
                double cls_conf = 0;
                int cls_idx = 0;
                minMaxIdx(classes, nullptr, &cls_conf, nullptr, &cls_idx);

                bboxes[cls_idx].push_back(rect);
                scores[cls_idx].push_back(cls_conf);
            }
        }
    }

    vector<BoundingBox> ret;
    for (size_t cls = 0; cls < bboxes.size(); ++cls) {
        vector<int> indices;
        NMSBoxes(bboxes[cls], scores[cls], threshold_, nms_threshold_, indices);
        for (size_t i = 0; i < indices.size(); ++i) {
            BoundingBox bbox;
            bbox.bbox = bboxes[cls][indices[i]];
            if (fix_bbox_) {
                bbox.bbox &= image_rect;
            }
            bbox.class_id = cls;
            bbox.confidence = scores[cls][indices[i]];
            ret.push_back(bbox);
        }
    }

    return ret;
}

void YoloOnnx::check_input_count() {
    size_t input_count = session_.GetInputCount();
    if (input_count != 1) {
        throw runtime_error("Expect input count == 1");
    }
}
