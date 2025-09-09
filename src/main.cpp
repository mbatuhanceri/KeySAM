#include <iostream>
#include <string>
#include "pipelines/HumanSegmenterPipeline.h"

int main() {
	//Configs
	std::string inputVideoPath = "D:\\Projects_after_april\\repos2\\peoplenet\\dataset\\videos\\4.mp4";
	std::string outputVideoPath = "D:\\Projects_after_april\\repos2\\peoplenet\\dataset\\videos\\4-output.mp4";

	// Pipeline Create
	HumanSegmenterPipeline humanSegmenterPipeline(inputVideoPath, outputVideoPath);
	std::cin >> inputVideoPath;
}