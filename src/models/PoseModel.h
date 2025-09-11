#pragma once

#include "IModel.h"
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>
#include <torch/script.h>

using torch::indexing::Slice;
using torch::indexing::None;

struct KeyPoint {
	cv::Point pt;
	float confidence;
};

class PoseModel : public IModel {
	public:
	torch::jit::Module* model_ = nullptr;

	PoseModel() {
		load();
	}
	~PoseModel() {
		delete model_;
		model_ = nullptr;
	}

	void load() override;

	std::string getName() override {
		return "PoseModel";
	}


	private:
	std::string model_path = "D:\\uzumaki_AI\\yolov8s-pose.torchscript";	//todo config' den alınmalı
	torch::Device device =  torch::kCUDA;

};