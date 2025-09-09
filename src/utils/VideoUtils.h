#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>


class VideoUtils {
public:
	VideoUtils() {};
	~VideoUtils() {
		close();
	};

	bool openVideo(const std::string& inputVideoPath) {
		close();

		if (!cap_.open(inputVideoPath)) {
			std::cerr << "Cannot open video file! " << inputVideoPath << std::endl;
			return false;
		}

		std::cout << "Video opened scucesfully!" << inputVideoPath << std::endl;
		return true;
	}

	bool isOpened() const {
		return cap_.isOpened();
	}

	void close() {
		if (cap_.isOpened()) {
			cap_.release();
			cv::destroyAllWindows();
		}
	}

	bool readFrame(cv::Mat& frame) {
		if (!cap_.isOpened()) {
			return false;
		}

		cap_ >> frame;
		return !frame.empty();
	}

	bool hasNextFrame() {
		if (!cap_.isOpened()) return false;

		cv::Mat testFrame;
		cap_ >> testFrame;

		if (!testFrame.empty()) {
			// Put frame back (approximate solution)
			cap_.set(cv::CAP_PROP_POS_FRAMES, cap_.get(cv::CAP_PROP_POS_FRAMES) - 1);
			return true;
		}
		return false;
	}

private:
	cv::VideoCapture cap_;
	cv::Mat frame;

};

