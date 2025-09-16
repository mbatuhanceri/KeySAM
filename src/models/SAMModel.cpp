
#include "SAMModel.h"
#include <string>
#include <filesystem>

void SAMModel::load() {
	try {
		std::string weightsPath = "D:\\uzumaki_AI\\sam_encoder_huge.pt";
		if (!std::filesystem::exists(weightsPath)) {
			std::cerr << "Model file not found: " << weightsPath << std::endl;
			return;
		}
		this->model_ = new torch::jit::Module(torch::jit::load(weightsPath, device));
		this->model_->eval();
		this->model_->to(device, torch::kFloat32);

	}
	catch (std::exception &e) {
		std::cerr << e.what() << std::endl;
	}

}
