
#include "PoseModelProcessor.h"


void PoseModelProcessor::doPredict() {
	// pre-process
	CVUtils::resizeMat(this->videoUtils->current_frame, this->videoUtils->preprocessedFrame,
		this->frameWidth, this->frameHeight);
	CVUtils::cvtColor(this->videoUtils->preprocessedFrame, this->videoUtils->preprocessedFrame);

	std::vector<torch::jit::IValue> inputs = TensorUtils::prepareImageForPredict(this->videoUtils->preprocessedFrame);
	std::cout << inputs[0].toTensor().sizes() << std::endl;

	// predict
	torch::Tensor output = this->poseModel->model_->forward(inputs).toTensor();
	std::cout << "output" <<  output.sizes() << std::endl;

	int batch_size = output.size(0);
	int num_channels = output.size(1); // 56
	int num_detections = output.size(2);

	std::cout << "Batch: " << batch_size
			  << ", Channels: " << num_channels
			  << ", Detections: " << num_detections << std::endl;

	// post-process


}
