#include "transforms/Transforms.hpp"

namespace Transforms {

void applyTranslateCPU(cv::Mat& frame, double dx, double dy) {
    if (frame.empty()) return;
    cv::Mat M = (cv::Mat_<double>(2, 3) << 1.0, 0.0, dx, 0.0, 1.0, dy);
    cv::warpAffine(frame, frame, M, frame.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(51, 25.5, 25.5));
}

void applyScaleCPU(cv::Mat& frame, double sx, double sy, double pivotX,
                   double pivotY) {
    if (frame.empty()) return;
    // If pivot not provided (negative), use image center
    double px = pivotX;
    double py = pivotY;
    if (px < 0.0 || py < 0.0) {
        px = frame.cols * 0.5;
        py = frame.rows * 0.5;
    }

    std::cout << "Scaling about pivot (" << px << ", " << py << ")\n";
    // Build affine: scale about pivot -> T(p) * S * T(-p)
    // Which yields matrix: [ sx 0  (1-sx)*px ; 0 sy (1-sy)*py ]
    double tx = (1.0 - sx) * px;
    double ty = (1.0 - sy) * py;
    cv::Mat M = (cv::Mat_<double>(2, 3) << sx, 0.0, tx, 0.0, sy, ty);
    cv::warpAffine(frame, frame, M, frame.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(51, 25.5, 25.5));
}

void applyRotateCPU(cv::Mat& frame, double angleDegrees) {
    if (frame.empty()) return;
    cv::Point2f center(frame.cols * 0.5f, frame.rows * 0.5f);
    cv::Mat M = cv::getRotationMatrix2D(center, angleDegrees, 1.0);
    cv::warpAffine(frame, frame, M, frame.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(51, 25.5, 25.5));
}

std::string gpuFragmentPathTransform() { return "gpu_transform.frag"; }

}  // namespace Transforms
