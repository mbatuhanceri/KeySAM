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
	PoseModel() {
		load();
	}
	~PoseModel() {
		delete pose_model;
		pose_model = nullptr;
	}

	void load() override;

	std::string getName() override {
		return "PoseModel";
	}

	std::vector<cv::Mat> infer(const cv::Mat& frame, const std::vector<cv::Point>& points) {
		// 1. frame’i tensor’e çevir
		// 2. keypointleri modele uygun formata dönüştür
		// 3. model forward çalıştır
		// 4. çıktı maskeleri dönüştür
		std::vector<cv::Mat> dummyMasks;
		// (şimdilik sahte maske üretelim)
		dummyMasks.push_back(cv::Mat::zeros(frame.size(), CV_8UC1));
		return dummyMasks;
	}

	private:
	torch::jit::Module* pose_model = nullptr;
	std::string model_path = "D:\\uzumaki_AI\\yolov8s-pose.torchscript";	//todo config' den alınmalı
	torch::Device device =  torch::kCUDA;
	std::vector<std::string> classes {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
									  "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
									  "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
									  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
									  "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
									  "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
									  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
};