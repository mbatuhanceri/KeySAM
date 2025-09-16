#pragma once

#include "HumanSegmenterPipeline.h"
#include "../utils/VideoUtils.h"
#include "../utils/CVUtils.h"

void HumanSegmenterPipeline::run() {
	this->videoUtils->openVideo(this->inputVideoPath);

	while (this->videoUtils->readFrame()) {
		this->poseModelProcessor->doPredict();
		this->poseModelProcessor->processDetections();

		//this->samModelProcessor->doPredict();
		//this->samModelProcessor->processDetections();


		CVUtils::showFrame(this->videoUtils->overlayedFrame, 1);	//opsiyonel
		//std::cout << counter++ << std::endl;
	}

}
