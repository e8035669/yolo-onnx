#ifndef _BOUNDING_BOX_H_
#define _BOUNDING_BOX_H_

#include <opencv2/core.hpp>

struct BoundingBox {
    cv::Rect bbox;
    int class_id;
    float confidence;
};

#endif
