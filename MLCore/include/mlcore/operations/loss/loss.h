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
	TensorCore::Tensor<T> BinaryCrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator);

	template <typename T>
	TensorCore::Tensor<T> CrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator);

	template <typename T>
	TensorCore::Tensor<T> CrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator);

	template <typename T>
	TensorCore::Tensor<T> CrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, size_t axis, Memory::ArenaAllocator& allocator);

	// Planning to add other loss functions (Mean Bias Error, Huber/Smooth Mean Absolute Error, etc.)
}

#include "loss.inl"