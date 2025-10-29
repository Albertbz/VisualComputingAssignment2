/*
 * Filters.hpp
 *
 * CPU + GPU filter placeholders for the webcam feed.
 * CPU functions operate on cv::Mat frames.
 * GPU helpers return fragment shader file paths (placeholders live in Webcam/).
 */
#ifndef FILTERS_HPP
#define FILTERS_HPP

#include <opencv2/opencv.hpp>
#include <string>

namespace Filters {

// CPU implementations (operate in-place on frames)
// - expect BGR 3-channel or BGRA 4-channel images coming from OpenCV
void applyGrayscaleCPU(cv::Mat& frame);
void applyCannyCPU(cv::Mat& frame, double threshold1 = 50.0,
                   double threshold2 = 150.0);
void applyPixelateCPU(cv::Mat& frame, int pixelSize = 10);

// GPU helpers: return path to fragment shader files that implement the
// corresponding GPU version of the filter (these are GLSL placeholders).
std::string gpuFragmentPathGrayscale();
std::string gpuFragmentPathEdge();
std::string gpuFragmentPathPixelate();

}  // namespace Filters

#endif
