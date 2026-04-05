// loss.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	template <typename T>
	TensorCore::Tensor<T> MeanSquaredError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator);

	template <typename T>
	TensorCore::Tensor<T> MeanAbsoluteError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator);

	template <typename T>
	TensorCore::Tensor<T> BinaryCrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator);

	template <typename T>
	TensorCore::Tensor<T> CrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator);
}

#include "loss.inl"