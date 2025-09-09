#pragma once

#include <opencv2/opencv.hpp>

class CVUtils {
public:
	static void showFrame(const cv::Mat& frame,
						 const std::string& windowName = "Frame") {
		if (frame.empty()) {
			std::cerr << "Error: Empty frame" << std::endl;
			return;
		}

		cv::imshow(windowName, frame);
		cv::waitKey(1);
	}

};

