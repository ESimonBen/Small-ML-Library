 /// linalg.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	/// <summary>
	/// Performs a standard dense matrix multiplication of two 2D tensors (A * B) and returns the resulting 2D tensor.
	/// </summary>
	/// <typeparam name="T">Element type of the input and output tensors; must support default construction, addition, and multiplication.</typeparam>
	/// <param name="A">Left operand tensor (must be 2D) with shape {M, K}. Elements are of type T. Treated as row-major linear storage.</param>
	/// <param name="B">Right operand tensor (must be 2D) with shape {K, N}. Elements are of type T. Treated as row-major linear storage.</param>
	/// <param name="allocator">Allocator used to construct and back the result tensor's storage.</param>
	/// <returns>A TensorCore::Tensor<T> with shape {M, N} containing the matrix product A * B. Throws std::runtime_error if either input is not rank 2 or if the inner dimensions (K) do not match. If A or B requires gradients, the result will be marked to require gradients and a MatMul gradient function will be attached.</returns>
	template <typename T>
	TensorCore::Tensor<T> MatMultiply(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Returns the 2D transpose of tensor A. Throws std::runtime_error if A.Rank() != 2. Allocates a new tensor using the provided allocator, copies elements with rows and columns swapped (row-major indexing), and if A.RequiresGrad() is true, marks the result as requiring gradients and attaches the corresponding transpose gradient function.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensor.</typeparam>
	/// <param name="A">Const reference to the input tensor to transpose. Must be rank 2.</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate storage for the returned tensor.</param>
	/// <returns>A new TensorCore::Tensor<T> with dimensions swapped (rows ↔ columns) containing the transposed elements. If the input required gradients, the returned tensor will also require gradients and have its gradient function set appropriately.</returns>
	template <typename T>
	TensorCore::Tensor<T> Transpose(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the dot product of two 1-dimensional tensors of the same size and returns the result as a scalar (1-element) tensor. If either input requires gradients, the result is marked to require gradients and a gradient function is attached. Throws if inputs are not 1D or their sizes differ.
	/// </summary>
	/// <typeparam name="T">Element type of the input tensors and of the resulting tensor.</typeparam>
	/// <param name="A">First input tensor. Must be 1-dimensional and have the same number of elements as B.</param>
	/// <param name="B">Second input tensor. Must be 1-dimensional and have the same number of elements as A.</param>
	/// <param name="allocator">Memory::ArenaAllocator used to construct/allocate the resulting tensor.</param>
	/// <returns>A TensorCore::Tensor<T> shaped as {1} whose single element is the dot product (sum of element-wise products). If either input required gradients, the returned tensor will require gradients and include an attached gradient function.</returns>
	template <typename T>
	TensorCore::Tensor<T> Dot(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator);
}

#include "linalg.inl"