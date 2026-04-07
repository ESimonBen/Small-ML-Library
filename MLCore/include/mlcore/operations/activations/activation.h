// activation.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	template <typename T>
	TensorCore::Tensor<T> ReLU(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	TensorCore::Tensor<T> LeakyReLU(const TensorCore::Tensor<T>& A, T alpha, Memory::ArenaAllocator& allocator);

	template <typename T>
	TensorCore::Tensor<T> Sigmoid(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	template <typename T>
	TensorCore::Tensor<T> Tanh(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	// Planning to add other activation functions (softmax mainly,s possibly others)
}

#include "activation.inl"