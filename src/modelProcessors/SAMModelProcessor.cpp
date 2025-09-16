#pragma once


#include "SAMModelProcessor.h"
#include <string>


void SAMModelProcessor::doPredict() {
	// her person için maske çıkartıp listeye ekle

	//preprocessing
//	CVUtils::resizeMat(this->videoUtils->current_frame, this->videoUtils->preprocessedFrame,
//						this->frameWidth, this->frameHeight);
//
//	CVUtils::converTo32f(this->videoUtils->preprocessedFrame, this->videoUtils->preprocessedFrame);
//
//	torch::Tensor img_tensor = torch::from_blob(this->videoUtils->preprocessedFrame.data, {1, 1024, 1024, 3},
//												torch::kFloat);
//	torch::Device device =  torch::kCUDA;
//	img_tensor = img_tensor.permute({0, 3, 1, 2}).to(device); // BHWC -> BCHW + GPU
//
//	std::vector<float> point_coords;
//	std::vector<float> point_labels;
//
//	for(const auto& pt : keypoints) {
//		point_coords.push_back(pt.x * 1024.0f / image.cols); // normalize
//		point_coords.push_back(pt.y * 1024.0f / image.rows);
//		point_labels.push_back(1.0f); // foreground point
//	}
//
//	torch::Tensor coords_tensor = torch::from_blob(point_coords.data(),
//		{1, (int)keypoints.size(), 2}, torch::kFloat).to(device);
//	torch::Tensor labels_tensor = torch::from_blob(point_labels.data(),
//		{1, (int)keypoints.size()}, torch::kFloat).to(device);
//
//
//	// Model inference (GPU'da)
//	std::vector<torch::jit::IValue> inputs;
//	inputs.push_back(img_tensor);
//	inputs.push_back(coords_tensor);
//	inputs.push_back(labels_tensor);
//
//	torch::Tensor mask = model.forward(inputs).toTensor();
//
//	// Sonucu CPU'ya geri getir (isteğe bağlı)
//	return mask.to(torch::kCPU);

}
