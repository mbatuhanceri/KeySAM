#pragma once

#include <opencv2/opencv.hpp>

class CVUtils {
public:
	static void showFrame(const cv::Mat& frame, int delay,
						 const std::string& windowName = "Frame") {
		if (frame.empty()) {
			std::cerr << "Error: Empty frame" << std::endl;
			return;
		}

		cv::imshow(windowName, frame);
		cv::waitKey(delay);		// 0 = wait
	}

	static void resizeMat(cv::Mat& frame, cv::Mat& targetFrame, int width, int height) {
		cv::resize(frame, targetFrame, cv::Size(width, height), cv::INTER_LINEAR);
	}

	static void cvtColor(cv::Mat& frame, cv::Mat& targetFrame) {
		cv::cvtColor(frame, targetFrame, cv::COLOR_BGR2RGB);
	}

};

