#include <opencv2/opencv.hpp>
#include <iostream>

#include "utils/ConfigManager.h"

int main() {


	ConfigManager config("D:\\CPP_PROJECTS\\keypoint-sam-pipeline\\config\\config.json");
	config.printConfig();
	cv::Mat image = cv::imread("D:\\test_sam2.jpg", cv::IMREAD_COLOR);

	return 0;
}
