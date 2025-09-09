#pragma once

#include <string>
#include "IPipeline.h"
#include "../modelProcessors/PoseModelProcessor.h"

class HumanSegmenterPipeline : public IPipeline{
	public:
	HumanSegmenterPipeline(const std::string inputVideoPath, const std::string outputVideoPath) : inputVideoPath(inputVideoPath),
		outputVideoPath(outputVideoPath) {
		poseModelProcessor = new PoseModelProcessor();

		run();
	};

	private:
	PoseModelProcessor* poseModelProcessor = nullptr;
	std::string inputVideoPath;
	std::string outputVideoPath;

	void run() override;

};

