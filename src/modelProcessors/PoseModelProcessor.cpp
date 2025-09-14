
#include "PoseModelProcessor.h"


void PoseModelProcessor::doPredict() {
	// pre-process
	CVUtils::resizeMat(this->videoUtils->current_frame, this->videoUtils->preprocessedFrame,
		this->frameWidth, this->frameHeight);
	CVUtils::cvtColor(this->videoUtils->preprocessedFrame, this->videoUtils->preprocessedFrame);

	std::vector<torch::jit::IValue> inputs = TensorUtils::prepareImageForPredict(this->videoUtils->preprocessedFrame);
	std::cout << inputs[0].toTensor().sizes() << std::endl;

	// predict
	this->modelOutput = this->poseModel->model_->forward(inputs).toTensor().to(torch::kCPU);
	std::cout << "output" <<  this->modelOutput.sizes() << std::endl;


	int batch_size = output.size(0);
	int num_channels = output.size(1); // 56
	int num_detections = output.size(2);

	std::cout << "Batch: " << batch_size
			  << ", Channels: " << num_channels
			  << ", Detections: " << num_detections << std::endl;

void PoseModelProcessor::processDetections() {
	float conf_threshold = 0.5;	//todo. must taken from config
	float nms_threshold = 0.5f;

	std::vector<Person> detections;
	// [1, 56, 8400] -> [batch, 4(bbox) + 1 (conf) + 51 (17 keyoints * 3(w,h,conf))]

	auto data = this->modelOutput.accessor<float, 3>();
	std::vector<torch::Tensor> boxes_list;
	std::vector<torch::Tensor> scores_list;

	for (int i = 0; i < 8400; i++) {
		float conf = data[0][4][i];
		if (conf < conf_threshold) continue;

		float cx = data[0][0][i];
		float cy = data[0][1][i];
		float w  = data[0][2][i];
		float h  = data[0][3][i];

		// Center format → xyxy format
		float x1 = cx - w / 2.0f;
		float y1 = cy - h / 2.0f;
		float x2 = cx + w / 2.0f;
		float y2 = cy + h / 2.0f;

		// Box + score tensörleri
		boxes_list.push_back(torch::tensor({x1, y1, x2, y2}));
		scores_list.push_back(torch::tensor(conf));

		// Person struct doldur
		Person person;
		person.x = cx;
		person.y = cy;
		person.width = w;
		person.height = h;
		person.confidence = conf;

		person.keypoints.resize(17);
		for (int j = 0; j < 17; j++) {
			person.keypoints[j].x = data[0][5 + j * 3][i];
			person.keypoints[j].y = data[0][5 + j * 3 + 1][i];
			person.keypoints[j].confidence = data[0][5 + j * 3 + 2][i];
		}

		detections.push_back(person);
	}

	if (detections.empty()) {
		std::cout << "Hic insan bulunamadı" << std::endl;
		return;
	}

	// Tensor stack
	auto boxes  = torch::stack(boxes_list).to(torch::kFloat32);
	auto scores = torch::stack(scores_list).to(torch::kFloat32);

	// torchvision::ops::nms -> keep indices
	auto keep = vision::ops::nms(boxes, scores, nms_threshold);

	std::vector<Person> finalDetections;
	for (int i = 0; i < keep.size(0); i++) {
		int idx = keep[i].item<int>();
		finalDetections.push_back(detections[idx]);
	}

	std::cout << finalDetections.size() << " insan tespit edildi" << std::endl;
}