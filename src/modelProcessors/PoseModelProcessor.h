#pragma once
#include "../models/PoseModel.h"
#include "../utils/TensorUtils.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <torchvision/ops/nms.h>

#include "../utils/VideoUtils.h"
#include "../utils/CVUtils.h"


class PoseModelProcessor {
private:
	struct KeyPoint {
		float x;
		float y;
		float confidence;
	};

	struct Person {
		float x, y, width, height;  // bounding box
		float confidence;           // detection confidence
		std::vector<KeyPoint> keypoints;  // 17 keypoints for COCO pose
	};

public:
	PoseModelProcessor(VideoUtils* videoUtils, int frameWidth, int frameHeight) : videoUtils(videoUtils), frameWidth(frameWidth),
		frameHeight(frameHeight) {
		poseModel = new PoseModel();
		std::cout << poseModel->getName() << " model is loaded!" << std::endl;
	}

	void doPredict();
	void processDetections();

private:
	VideoUtils* videoUtils;
	PoseModel* poseModel;

	torch::Tensor modelOutput;

	int frameWidth, frameHeight;



};
