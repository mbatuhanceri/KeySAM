#pragma once

#include "HumanSegmenterPipeline.h"
#include "../utils/VideoUtils.h"
#include "../utils/CVUtils.h"

void HumanSegmenterPipeline::run() {
	this->videoUtils->openVideo(this->inputVideoPath);

	int counter = 0;
	while (this->videoUtils->readFrame()) {
		this->poseModelProcessor->doPredict();
		this->poseModelProcessor->processDetections();


		CVUtils::showFrame(this->videoUtils->preprocessedFrame, 1);	//opsiyonel
		std::cout << counter++ << std::endl;
	}

}
