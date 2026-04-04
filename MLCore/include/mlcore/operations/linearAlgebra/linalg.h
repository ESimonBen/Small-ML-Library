// linalg.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	template <typename T>
	TensorCore::Tensor<T> MatMultiply(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator);

	template <typename T>
	TensorCore::Tensor<T> Transpose(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	TensorCore::Tensor<T> Dot(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator);

	template <typename T>
	TensorCore::Tensor<T> AxisSum(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator);
}

#include "linalg.inl"