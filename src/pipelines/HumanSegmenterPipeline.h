#pragma once

#include <string>
#include "IPipeline.h"
#include "../modelProcessors/PoseModelProcessor.h"
#include "../modelProcessors/SAMModelProcessor.h"
#include "../utils/VideoUtils.h"


class HumanSegmenterPipeline : public IPipeline{
public:
	HumanSegmenterPipeline(const std::string inputVideoPath, const std::string outputVideoPath) : inputVideoPath(inputVideoPath),
		outputVideoPath(outputVideoPath) {
		videoUtils = new VideoUtils();
		poseModelProcessor = new PoseModelProcessor(videoUtils, 640, 640);	//todo. get frame details from config
		samModelProcessor = new SAMModelProcessor(videoUtils, 1024, 1024);	//todo. get frame details from config
		run();
	};

private:
	VideoUtils* videoUtils;
	PoseModelProcessor* poseModelProcessor = nullptr;
	SAMModelProcessor* samModelProcessor = nullptr;
	std::string inputVideoPath;
	std::string outputVideoPath;

	void run() override;

};

