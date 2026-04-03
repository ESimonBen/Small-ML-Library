// broadcast.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	struct BroadcastInfo {
		Utils::Shape shape;
		std::vector<size_t> strideA;
		std::vector<size_t> strideB;
	};

	[[nodiscard]] BroadcastInfo ComputeBroadcast(const Utils::Shape& shapeA, const Utils::Shape& shapeB);
	[[nodiscard]] bool CanBroadcast(const Utils::Shape& shapeA, const Utils::Shape& shapeB);
}