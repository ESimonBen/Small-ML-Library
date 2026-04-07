// reduction.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	template <typename T>
	[[nodiscard]] T Sum(const TensorCore::Tensor<T>& A);

	template <typename T>
	[[nodiscard]] T Mean(const TensorCore::Tensor<T>& A);

	template <typename T>
	[[nodiscard]] T Max(const TensorCore::Tensor<T>& A);

	template <typename T>
	[[nodiscard]] T Min(const TensorCore::Tensor<T>& A);

	template <typename T>
	TensorCore::Tensor<T> AxisSum(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator);
}

#include "reduction.inl"