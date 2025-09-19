#pragma once


#include "SAMModelProcessor.h"
#include <string>


void SAMModelProcessor::doPredict() {
	/*
	try {
		//preprocessing
		CVUtils::resizeMat(this->videoUtils->current_frame, this->videoUtils->preprocessedFrame,
							this->frameWidth, this->frameHeight);

		CVUtils::converTo32f(this->videoUtils->preprocessedFrame, this->videoUtils->preprocessedFrame);

		torch::Tensor img_tensor = torch::from_blob(this->videoUtils->preprocessedFrame.data, {1, 1024, 1024, 3},
													torch::kFloat);
		torch::Device device =  torch::kCUDA;
		img_tensor = img_tensor.permute({0, 3, 1, 2}).to(device); // BHWC -> BCHW + GPU

		std::vector<float> point_coords;
		std::vector<float> point_labels;
		if (this->poseDetections.empty()) {
			std::cout << "11111!!!" << std::endl;
			return;
		};
		std::vector<Person> persons = this->poseDetections;


		for (const auto& person : persons) {
			std::vector<int> pointsWillBeUsed = {0, 5, 6, 11, 12};

			pointsWillBeUsed.erase(
				std::remove_if(pointsWillBeUsed.begin(), pointsWillBeUsed.end(),
					[&](int idx) {
						return person.keypoints[idx].confidence < 0.8;
					}),
				pointsWillBeUsed.end()
			);

			if (pointsWillBeUsed.empty()) break;
			for (int pt : pointsWillBeUsed) {
				point_coords.push_back(person.keypoints[pt].x);
				point_coords.push_back(person.keypoints[pt].y);
				point_labels.push_back(1.0f);
			}

			torch::Tensor coords_tensor = torch::from_blob(point_coords.data(),
				{1, (int)pointsWillBeUsed.size(), 2}, torch::kFloat).to(device);
			torch::Tensor labels_tensor = torch::from_blob(point_labels.data(),
				{1, (int)pointsWillBeUsed.size()}, torch::kFloat).to(device);


			std::vector<torch::jit::IValue> inputs;
			inputs.push_back(img_tensor);
			inputs.push_back(coords_tensor);
			inputs.push_back(labels_tensor);

			torch::Tensor mask = this->samModel->model_->forward(inputs).toTensor().to(torch::kCPU);
			std::cout << mask.sizes() << std::endl;
			std::cout << "SAM worked" << std::endl;
		}

	} catch (const std::exception& e) {
		std::cout << e.what() << std::endl;
	}

*/
}


void SAMModelProcessor::processDetections() {
	return;
}