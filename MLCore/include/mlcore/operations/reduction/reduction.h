// reduction.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> SumAll(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> MeanAll(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> MaxAll(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> MinAll(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	TensorCore::Tensor<T> AxisSum(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator, bool keepDims = false);

	template <typename T>
	TensorCore::Tensor<T> AxisMean(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator, bool keepDims = false);

	template <typename T>
	TensorCore::Tensor<T> AxisMax(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator, bool keepDims = false);

	template <typename T>
	TensorCore::Tensor<T> AxisMin(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator, bool keepDims = false);
}

#include "reduction.inl"
