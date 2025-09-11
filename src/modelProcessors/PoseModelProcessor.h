#pragma once
#include "../models/PoseModel.h"
#include "../utils/TensorUtils.h"

#include <torch/torch.h>
#include <torch/script.h>

#include "../utils/VideoUtils.h"
#include "../utils/CVUtils.h"

class PoseModelProcessor {
public:
	PoseModelProcessor(VideoUtils* videoUtils, int frameWidth, int frameHeight) : videoUtils(videoUtils), frameWidth(frameWidth),
		frameHeight(frameHeight) {
		poseModel = new PoseModel();
		std::cout << poseModel->getName() << " model is loaded!" << std::endl;
	}

	void doPredict();

private:
	VideoUtils* videoUtils;
	PoseModel* poseModel;

	int frameWidth, frameHeight;



};
