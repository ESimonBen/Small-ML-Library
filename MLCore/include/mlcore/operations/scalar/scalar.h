// scalar.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> AddScalar(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator)  noexcept;

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> MultiplyScalar(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator) noexcept;

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> SubtractScalar(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator, bool scalarOnLeft) noexcept;

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> DivideScalar(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator, bool scalarOnLeft);
}

#include "scalar.inl"