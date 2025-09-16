#pragma once

#include <vector>


struct KeyPoint {
	float x;
	float y;
	float confidence;
};

struct Person {
	float x, y, width, height;
	float confidence;
	std::vector<KeyPoint> keypoints;
};

class MathUtils {
public:
	static std::vector<Person> scaleKeyPoints(const std::vector<Person>& data, int frameWidth, int frameHeight) {
		std::vector<Person> scaledData;
		scaledData.reserve(data.size());

		for (const auto& person : data) {
			Person scaledPerson;

			// Bbox
			scaledPerson.x = person.x * frameWidth;
			scaledPerson.y = person.y * frameHeight;
			scaledPerson.width = person.width * frameWidth;
			scaledPerson.height = person.height * frameHeight;
			scaledPerson.confidence = person.confidence;

			// Keypoints
			for (const auto& kp : person.keypoints) {
				KeyPoint scaledKP;
				scaledKP.x = kp.x * frameWidth;
				scaledKP.y = kp.y * frameHeight;
				scaledKP.confidence = kp.confidence;
				scaledPerson.keypoints.push_back(scaledKP);
			}

			scaledData.push_back(std::move(scaledPerson));
		}

		return scaledData;
	}

};
