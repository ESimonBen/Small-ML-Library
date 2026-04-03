// scalar.h
#pragma once
#include <concepts>
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	// Scalar Operations on RHS
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> AddScalarRight(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator)  noexcept;

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> SubtractScalarRight(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator) noexcept;

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> MultiplyScalarRight(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator) noexcept;

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> DivideScalarRight(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator);

	// Scalar Operations on LHS
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> AddScalarLeft(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator)  noexcept;

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> SubtractScalarLeft(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator) noexcept;

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> MultiplyScalarLeft(const TensorCore::Tensor<T>& A, const T Scalar, Memory::ArenaAllocator& allocator) noexcept;

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> DivideScalarLeft(const TensorCore::Tensor<T>& A, const T Scalar, Memory::ArenaAllocator& allocator);
}

#include "scalar.inl"