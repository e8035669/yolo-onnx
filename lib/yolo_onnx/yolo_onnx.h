#ifndef _YOLO_ONNX_H_
#define _YOLO_ONNX_H_

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <memory>
#include <opencv2/core/core.hpp>

#include "bounding_box.h"

class YoloOnnx {
   public:
    YoloOnnx() : YoloOnnx(nullptr) {}
    YoloOnnx(Ort::Env* env)
        : env_(env),
          allocator_(),
          session_(nullptr),
          output_name_("output"),
          blob_(),
          threshold_(0.5),
          nms_threshold_(0.5),
          fix_bbox_(true) {}

    void set_env(Ort::Env* env) { env_ = env; }

    std::string output_name() { return output_name_; }
    void set_output_name(const std::string& output_name) {
        output_name_ = output_name;
    }

    float threshold() { return threshold_; }
    void set_threshold(float threshold) { threshold_ = threshold; }

    float nms_threshold() { return nms_threshold_; }
    void set_nms_threshold(float nms_threshold) {
        nms_threshold_ = nms_threshold;
    }

    bool fix_bbox() { return fix_bbox_; }
    void set_fix_bbox(bool fix_bbox) { fix_bbox_ = fix_bbox; }

    void load_model(const std::string& filename);

    const char* get_input_name();

    std::vector<int64_t> get_input_shape();

    std::vector<BoundingBox> detect(cv::InputArray image);

   private:
    Ort::Env* env_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::Session session_;
    std::string output_name_;

    cv::Mat blob_;

    float threshold_;
    float nms_threshold_;
    bool fix_bbox_;

    void check_input_count();
};

#endif
