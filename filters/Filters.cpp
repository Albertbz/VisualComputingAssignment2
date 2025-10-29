#include "Filters.hpp"

namespace Filters {

void applyGrayscaleCPU(cv::Mat& frame) {
    if (frame.empty()) return;
    // Convert to grayscale then back to BGR to keep 3 channels (texture code
    // expects BGR data in this project).
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(gray, frame, cv::COLOR_GRAY2BGR);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
        cv::cvtColor(gray, frame, cv::COLOR_GRAY2BGR);
    } else if (frame.channels() == 1) {
        // Already single channel
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    }
}

void applyCannyCPU(cv::Mat& frame, double threshold1, double threshold2) {
    if (frame.empty()) return;
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = frame.clone();
    }

    cv::Mat edges;
    cv::Canny(gray, edges, threshold1, threshold2);

    // Convert edges -> BGR so the rest of the pipeline (which expects 3
    // channels) continues to work.
    cv::Mat edgesBGR;
    cv::cvtColor(edges, edgesBGR, cv::COLOR_GRAY2BGR);
    frame = edgesBGR;
}

void applyPixelateCPU(cv::Mat& frame, int pixelSize) {
    if (frame.empty() || pixelSize <= 1) return;

    cv::Mat temp = frame.clone();
    for (int y = 0; y < frame.rows; y += pixelSize) {
        for (int x = 0; x < frame.cols; x += pixelSize) {
            // Define the region of interest
            int width = std::min(pixelSize, frame.cols - x);
            int height = std::min(pixelSize, frame.rows - y);
            cv::Rect roi(x, y, width, height);

            // Compute the average color in the ROI
            cv::Scalar avgColor = cv::mean(temp(roi));

            // Fill the ROI with the average color
            frame(roi).setTo(avgColor);
        }
    }
}

std::string gpuFragmentPathGrayscale() { return "gpu_grayscale.frag"; }

std::string gpuFragmentPathEdge() { return "gpu_edge.frag"; }

std::string gpuFragmentPathPixelate() { return "gpu_pixelate.frag"; }

}  // namespace Filters
