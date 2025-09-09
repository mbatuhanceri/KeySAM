
#include "PoseModel.h"
#include <string>
#include <filesystem>

//void PoseModel::load() {
//	try {
//		//Todo. config instance oluşturulmalı
//		std::string weightsPath = "D:\\uzumaki_AI\\yolov8s-pose.torchscript";
//		pose_model = new torch::jit::Module(torch::jit::load(weightsPath, device));
//		pose_model->eval();
//		pose_model->to(device, torch::kFloat32);
//	} catch (std::exception &e) {
//		std::cerr << e.what() << std::endl;
//	}
//}

void PoseModel::load() {
	try {
		//Todo. config instance oluşturulmalı
		std::string weightsPath = "D:\\uzumaki_AI\\yolov8s-pose.torchscript";
		if (!std::filesystem::exists(weightsPath)) {
			std::cerr << "Model file not found: " << weightsPath << std::endl;
			return;
		}
		pose_model = new torch::jit::Module(torch::jit::load(weightsPath, device));
		pose_model->eval();
		pose_model->to(device, torch::kFloat32);
	} catch (std::exception &e) {
		std::cerr << e.what() << std::endl;
	}
}
