#pragma once

#include "../models/PoseModel.h"
#include "../utils/TensorUtils.h"
#include "../utils/VideoUtils.h"
#include "../utils/MathUtils.h"
#include <torch/script.h>
#include <torchvision/ops/nms.h>


class IModelProcessors {
public:
	virtual ~IModelProcessors() = default;

	virtual void doPredict() = 0;
	virtual void processDetections() = 0;

	std::vector<Person> poseDetections;
	torch::Tensor poseOutput;
	torch::Tensor samOutput;

};

