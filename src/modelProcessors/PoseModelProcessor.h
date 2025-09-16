#pragma once

#include "IModelProcessors.h"



class PoseModelProcessor : public IModelProcessors{
public:
	PoseModelProcessor(VideoUtils* videoUtils, int frameWidth, int frameHeight) : videoUtils(videoUtils), frameWidth(frameWidth),
		frameHeight(frameHeight) {
		poseModel = new PoseModel();
		std::cout << poseModel->getName() << " model is loaded!" << std::endl;
	}

	void doPredict() override;
	void processDetections() override;
	void drawPoseDetections(const std::vector<Person>& finalDetections);


private:
	VideoUtils* videoUtils;
	PoseModel* poseModel;

	int frameWidth, frameHeight;

};
