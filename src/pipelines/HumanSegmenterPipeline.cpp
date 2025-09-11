#pragma once

#include "HumanSegmenterPipeline.h"
#include "../utils/VideoUtils.h"
#include "../utils/CVUtils.h"

void HumanSegmenterPipeline::run() {

	this->videoUtils->openVideo(this->inputVideoPath);

	while (this->videoUtils->readFrame()) {
		this->poseModelProcessor->doPredict();


		CVUtils::showFrame(this->videoUtils->preprocessedFrame, 1);	//opsiyonel
	}

}
