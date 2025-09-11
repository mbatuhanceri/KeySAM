#pragma once
#include <opencv2/core.hpp>
#include <torch/torch.h>
#include <torch/script.h>

using torch::indexing::Slice;
using torch::indexing::None;

class TensorUtils {
public:
	static std::vector<torch::jit::IValue> prepareImageForPredict(cv::Mat image) {
		torch::Device device =  torch::kCUDA;
		torch::Tensor imageTensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte).to(device);
		imageTensor = imageTensor.toType(torch::kFloat32).div(255);
		imageTensor = imageTensor.permute({2, 0, 1});
		imageTensor = imageTensor.unsqueeze(0);
		std::vector<torch::jit::IValue> inputs {imageTensor};
		return inputs;
	}

};
