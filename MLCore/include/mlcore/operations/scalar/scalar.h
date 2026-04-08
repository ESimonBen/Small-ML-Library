// scalar.h
#pragma once
#include <concepts>
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> AddScalar(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator)  noexcept;

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> MultiplyScalar(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator) noexcept;

	// Scalar Operations on RHS

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> SubtractScalarRight(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator) noexcept;

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> DivideScalarRight(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator);

	// Scalar Operations on LHS

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> SubtractScalarLeft(const T Scalar, const TensorCore::Tensor<T>& Input, Memory::ArenaAllocator& allocator) noexcept;

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> DivideScalarLeft(const T Scalar, const TensorCore::Tensor<T>& Input, Memory::ArenaAllocator& allocator);
}

#include "scalar.inl"