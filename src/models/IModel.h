#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <string>

class IModel {
public:
	virtual ~IModel() = default;

	virtual void load() = 0;

	virtual std::string getName() = 0;
};


