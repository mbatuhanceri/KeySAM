#pragma once

#include <string>
#include "IPipeline.h"
#include "../modelProcessors/PoseModelProcessor.h"
#include "../utils/VideoUtils.h"


class HumanSegmenterPipeline : public IPipeline{
public:
	HumanSegmenterPipeline(const std::string inputVideoPath, const std::string outputVideoPath) : inputVideoPath(inputVideoPath),
		outputVideoPath(outputVideoPath) {
		videoUtils = new VideoUtils();
		poseModelProcessor = new PoseModelProcessor(videoUtils, 640, 640);	//todo. get frame details from config
		run();
	};

private:
	VideoUtils* videoUtils;
	PoseModelProcessor* poseModelProcessor = nullptr;
	std::string inputVideoPath;
	std::string outputVideoPath;

	void run() override;

};

