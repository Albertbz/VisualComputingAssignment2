/*
 * Transforms.hpp
 *
 * CPU and GPU transform helpers (translate, scale, rotate).
 */
#ifndef TRANSFORMS_HPP
#define TRANSFORMS_HPP

#include <opencv2/opencv.hpp>
#include <string>

namespace Transforms {

// CPU implementations (operate in-place on cv::Mat frames)
// pivotX/pivotY (in pixels) specify the point to scale about. If both
// are negative the function falls back to scaling about the image center.
void applyTranslateCPU(cv::Mat& frame, double dx, double dy);
void applyScaleCPU(cv::Mat& frame, double sx, double sy, double pivotX = -1.0,
                   double pivotY = -1.0);
void applyRotateCPU(cv::Mat& frame, double angleDegrees);

// GPU helper: return fragment shader path implementing UV-space transform
std::string gpuFragmentPathTransform();

}  // namespace Transforms

#endif
