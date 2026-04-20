// reduction.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Sum(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Mean(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Max(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Min(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	TensorCore::Tensor<T> AxisSum(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator);

	template <typename T>
	TensorCore::Tensor<T> AxisMean(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator);
}

#include "reduction.inl"