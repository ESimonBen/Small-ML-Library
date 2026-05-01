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
	[[nodiscard]] BroadcastInfo ComputeBroadcastTo(const Utils::Shape& input, const Utils::Shape& target);
	[[nodiscard]] bool CanBroadcast(const Utils::Shape& shapeA, const Utils::Shape& shapeB);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Squeeze(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Unsqueeze(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator);
}

#include "broadcast.inl"