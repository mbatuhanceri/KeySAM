#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>


class VideoUtils {
public:
	cv::Mat current_frame;
	cv::Mat preprocessedFrame;
	cv::Mat overlayedFrame;

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

	bool readFrame() {
		if (!cap_.isOpened()) {
			return false;
		}

		cap_ >> this->current_frame;
		return !this->current_frame.empty();
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

	int getVideoWidth() {
		return static_cast<int>(cap_.get((cv::CAP_PROP_FRAME_WIDTH)));
	}

	int getVideoHeight() {
		return static_cast<int>(cap_.get((cv::CAP_PROP_FRAME_HEIGHT)));
	}

private:
	cv::VideoCapture cap_;

};

