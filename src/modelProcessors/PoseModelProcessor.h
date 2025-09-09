#pragma once
#include "../models/PoseModel.h"

class PoseModelProcessor {
public:
	PoseModelProcessor() {
		poseModel = new PoseModel();
		std::cout << poseModel->getName() << " model is loaded!" << std::endl;
	}


private:
	PoseModel* poseModel;

};
