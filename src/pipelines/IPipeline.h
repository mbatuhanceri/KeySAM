#pragma once

class IPipeline {
	public:
	virtual ~IPipeline() = default;
	virtual void run() = 0;
};


