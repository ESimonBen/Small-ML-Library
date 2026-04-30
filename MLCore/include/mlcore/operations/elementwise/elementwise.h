// elementwise.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Add(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Subtract(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Multiply(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Divide(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Power(const TensorCore::Tensor<T>& A, T exponent, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Abs(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Clamp(const TensorCore::Tensor<T>& A, T min, T max, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Log(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Exp(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Equal(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Negate(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Square(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Reciprocal(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);
}

#include "elementwise.inl"