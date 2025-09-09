#pragma once

#include "HumanSegmenterPipeline.h"
#include "../utils/VideoUtils.h"
#include "../utils/CVUtils.h"

void HumanSegmenterPipeline::run() {

	// videoyu aç
	VideoUtils videoUtils;
	CVUtils cvUtils;
	videoUtils.openVideo(this->inputVideoPath);


	// while (videodan frame al)
	cv::Mat frame;
	while (videoUtils.readFrame(frame)) {


		// Posedetection modeline sok pointleri al

	}

}
