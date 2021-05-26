#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "yolo_onnx.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << argv[0] << " <model> <image> ..." << endl;
        return -1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "infer_model");
    YoloOnnx yolo(&env);

    yolo.load_model(argv[1]);

    namedWindow("Window", WINDOW_NORMAL);

    for (int i = 2; i < argc; ++i) {
        cout << "Load image: " << argv[i] << endl;
        Mat image = imread(argv[i]);
        vector<BoundingBox> bboxes = yolo.detect(image);

        for (auto& bbox : bboxes) {
            rectangle(image, bbox.bbox, {0, 0, 255}, 2);

            cout << "box: " << bbox.bbox << ", cls: " << bbox.class_id
                 << ", conf: " << bbox.confidence << endl;
        }

        imshow("Window", image);
        char key = waitKey();
        if (key == 27) {
            break;
        }
    }

    //
    return 0;
}
