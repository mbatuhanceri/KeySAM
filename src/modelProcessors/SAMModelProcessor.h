#pragma once

#include "IModelProcessors.h"
#include "../models/SAMModel.h"
#include "../utils/TensorUtils.h"
#include "../utils/VideoUtils.h"
#include "../utils/CVUtils.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <torchvision/ops/nms.h>

class SAMModelProcessor{
public:
	SAMModelProcessor(VideoUtils* videoUtils, int frameWidth, int frameHeight) :videoUtils(videoUtils),
		frameWidth(frameWidth), frameHeight(frameHeight) {
		samModel = new SAMModel();
		std::cout << samModel->getName() << " model is loaded!" << std::endl;
	}

	void doPredict();
	void processDetections();
	void drawPoseDetections();

private:
	VideoUtils* videoUtils;
	SAMModel* samModel;

	torch::Tensor modelOutput;
	int frameWidth, frameHeight;

};


