#include "PoseModelProcessor.h"


void PoseModelProcessor::doPredict() {
	//preprocessing
	std::vector<torch::jit::IValue> inputs = TensorUtils::prepareImageForPosePredict(this->videoUtils->current_frame,
		this->videoUtils->preprocessedFrame, this->frameWidth, this->frameHeight);

	// predict
	this->poseOutput = this->poseModel->model_->forward(inputs).toTensor().to(torch::kCPU);
}

void PoseModelProcessor::processDetections() {
	float conf_threshold = 0.6; //todo. must taken from config
	float nms_threshold = 0.6f;

	std::vector<Person> detections;
	// [1, 56, 8400] -> [batch, 4(bbox) + 1 (conf) + 51 (17 keyoints * 3(w,h,conf))]

	auto data = this->poseOutput.accessor<float, 3>();
	std::vector<torch::Tensor> boxes_list;
	std::vector<torch::Tensor> scores_list;

	for (int i = 0; i < 8400; i++) {
		float conf = data[0][4][i];
		if (conf < conf_threshold) continue;

		float cx = data[0][0][i];
		float cy = data[0][1][i];
		float w = data[0][2][i];
		float h = data[0][3][i];

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
		person.x = cx / this->frameWidth;
		person.y = cy / this->frameHeight;
		person.width = w / this->frameWidth;
		person.height = h / this->frameHeight;
		person.confidence = conf;

		person.keypoints.resize(17);
		for (int j = 0; j < 17; j++) {
			person.keypoints[j].x = (data[0][5 + j * 3][i]) / this->frameWidth;
			person.keypoints[j].y = (data[0][5 + j * 3 + 1][i]) / this->frameHeight; //normalize
			person.keypoints[j].confidence = data[0][5 + j * 3 + 2][i];
		}

		detections.push_back(person);
	}

	if (detections.empty()) {
		std::cout << "No person detected!" << std::endl;
		return;
	}

	// Tensor stack
	auto boxes = torch::stack(boxes_list).to(torch::kFloat32);
	auto scores = torch::stack(scores_list).to(torch::kFloat32);

	// torchvision::ops::nms -> keep indices
	auto keep = vision::ops::nms(boxes, scores, nms_threshold);

	finalDetections.clear();
	for (int i = 0; i < keep.size(0); i++) {
		int idx = keep[i].item<int>();
		finalDetections.push_back(detections[idx]);
	}

	this->poseDetections = finalDetections;
	std::cout << finalDetections.size() << " person detected" << std::endl;

	if (true) {
		// will be taken from configs ( if(configs.getInstance().getDebugMode == true) {} )
		this->drawPoseDetections(finalDetections);
	}
}

void PoseModelProcessor::drawPoseDetections(const std::vector<Person> &finalDetections) {
	this->videoUtils->overlayedFrame = this->videoUtils->current_frame.clone();
	std::vector<Person> scaledDetections = MathUtils::scaleKeyPoints(finalDetections, this->videoUtils->getVideoWidth(),
																	this->videoUtils->getVideoHeight());

	// COCO pose keypoint connections (17 keypoints için)
	std::vector<std::pair<int, int> > pose_connections = {
		{0, 1}, {0, 2}, {1, 3}, {2, 4}, // Head
		{5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, // Arms
		{5, 11}, {6, 12}, {11, 12}, // Torso
		{11, 13}, {13, 15}, {12, 14}, {14, 16} // Legs
	};

	std::vector<cv::Scalar> keypoint_colors = {
		cv::Scalar(0, 0, 255), // 0: nose - red
		cv::Scalar(255, 0, 0), // 1: left_eye - blue
		cv::Scalar(0, 255, 0), // 2: right_eye - green
		cv::Scalar(255, 255, 0), // 3: left_ear - cyan
		cv::Scalar(255, 0, 255), // 4: right_ear - magenta
		cv::Scalar(0, 255, 255), // 5: left_shoulder - yellow
		cv::Scalar(128, 0, 128), // 6: right_shoulder - purple
		cv::Scalar(255, 128, 0), // 7: left_elbow - orange
		cv::Scalar(0, 128, 255), // 8: right_elbow - light blue
		cv::Scalar(128, 255, 0), // 9: left_wrist - lime
		cv::Scalar(255, 0, 128), // 10: right_wrist - pink
		cv::Scalar(64, 128, 255), // 11: left_hip - light orange
		cv::Scalar(255, 64, 128), // 12: right_hip - light pink
		cv::Scalar(128, 255, 128), // 13: left_knee - light green
		cv::Scalar(255, 128, 128), // 14: right_knee - light red
		cv::Scalar(64, 255, 255), // 15: left_ankle - light cyan
		cv::Scalar(255, 255, 64) // 16: right_ankle - light yellow
	};

	float keypoint_conf_threshold = 0.5f; // Keypoint confidence threshold

	for (const auto &person: scaledDetections) {

		for (const auto &connection: pose_connections) {
			int idx1 = connection.first;
			int idx2 = connection.second;

			if (idx1 < person.keypoints.size() && idx2 < person.keypoints.size()) {
				const auto &kp1 = person.keypoints[idx1];
				const auto &kp2 = person.keypoints[idx2];

				// Her iki keypoint de yeterli confidence'a sahipse çizgi çiz
				if (kp1.confidence > keypoint_conf_threshold &&
					kp2.confidence > keypoint_conf_threshold) {
					cv::line(this->videoUtils->overlayedFrame,
							cv::Point(static_cast<int>(kp1.x),
									static_cast<int>(kp1.y)),
							cv::Point(static_cast<int>(kp2.x),
									static_cast<int>(kp2.y)),
							cv::Scalar(255, 255, 255), 2);
				}
			}
		}

		for (int i = 0; i < person.keypoints.size() && i < keypoint_colors.size(); i++) {
			const auto &kp = person.keypoints[i];

			if (kp.confidence > keypoint_conf_threshold) {
				cv::Point center(static_cast<int>(kp.x),
								static_cast<int>(kp.y));

				cv::circle(this->videoUtils->overlayedFrame, center, 5, keypoint_colors[i], -1);
				cv::putText(this->videoUtils->overlayedFrame, std::to_string(i),
							cv::Point(center.x + 7, center.y - 7), cv::FONT_HERSHEY_SIMPLEX,
							0.3, cv::Scalar(255, 255, 255), 1);
			}
		}
	}
}
