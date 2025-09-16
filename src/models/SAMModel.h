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

class SAMModel : public IModel {
public:
	SAMModel() {
		load();
	}

	~SAMModel() {
		delete model_;
		model_ = nullptr;
	}

	torch::jit::Module* model_ = nullptr;

	void load() override;

	std::string getName() override {
		return "SAMModel";
	}


private:
	torch::Device device =  torch::kCUDA;
};

