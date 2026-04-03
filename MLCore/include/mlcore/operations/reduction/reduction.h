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
}

#include "reduction.inl"