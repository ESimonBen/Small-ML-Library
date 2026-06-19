 /// activation.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	/// <summary>
	/// Applies the Rectified Linear Unit (ReLU) activation element-wise to the input tensor.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor (e.g., float, double).</typeparam>
	/// <param name="A">Input tensor whose elements will be replaced by max(0, value).</param>
	/// <param name="allocator">Arena allocator used to allocate memory for the returned tensor.</param>
	/// <returns>A new TensorCore::Tensor<T> with the same shape as A containing ReLU(A). If A.RequiresGrad() is true, the returned tensor will require gradients and its gradient function will be set to the corresponding ReLU gradient function.</returns>
	template <typename T>
	TensorCore::Tensor<T> ReLU(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Applies the Leaky ReLU activation elementwise to an input tensor, returning a new tensor with the same shape. Positive values are kept; negative values are scaled by alpha. If the input requires gradients, the result will also require gradients and have an appropriate gradient function attached.
	/// </summary>
	/// <typeparam name="T">The numeric element type of the tensor (e.g., float, double).</typeparam>
	/// <param name="A">The input tensor to transform.</param>
	/// <param name="alpha">Slope for negative input values (multiplier applied to elements <= 0).</param>
	/// <param name="allocator">Memory allocator used to construct the output tensor.</param>
	/// <returns>A new Tensor<T> with the same shape as A containing the Leaky ReLU result for each element. If A.RequiresGrad() is true, the returned tensor will require gradients and have its gradient function set.</returns>
	template <typename T>
	TensorCore::Tensor<T> LeakyReLU(const TensorCore::Tensor<T>& A, T alpha, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the element-wise sigmoid activation 1 / (1 + exp(-x)) for each element of the input tensor.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor (e.g., float, double) used for computations.</typeparam>
	/// <param name="A">Input tensor whose elements will be transformed by the sigmoid function.</param>
	/// <param name="allocator">Memory allocator used to allocate intermediate tensors and the returned result.</param>
	/// <returns>A new tensor with the same shape as A where each element is the sigmoid of the corresponding element in A, allocated using the provided allocator.</returns>
	template <typename T>
	TensorCore::Tensor<T> Sigmoid(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the element-wise hyperbolic tangent (tanh) of the input tensor.
	/// </summary>
	/// <typeparam name="T">The numeric type of the tensor elements (e.g., float, double).</typeparam>
	/// <param name="A">Input tensor whose elements will be transformed by tanh.</param>
	/// <param name="allocator">Memory arena allocator used for intermediate allocations and for allocating the returned tensor.</param>
	/// <returns>A new tensor where each element is tanh of the corresponding element in A, computed as (exp(x) - exp(-x)) / (exp(x) + exp(-x)).</returns>
	template <typename T>
	TensorCore::Tensor<T> Tanh(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the softmax of all elements in the input tensor using a numerically stable formulation and returns a new tensor with the same shape whose elements sum to 1. (Should not be used in actual machine learning contexts, debugging / teaching moment for me only)
	/// </summary>
	/// <typeparam name="T">Element type of the tensor (e.g., float or double).</typeparam>
	/// <param name="A">Input tensor whose elements are used to compute the softmax. The operation is applied over all elements (flattened tensor). If A.RequiresGrad() is true, the returned tensor will be configured to propagate gradients.</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate storage for the result tensor.</param>
	/// <returns>A TensorCore::Tensor<T> with the same shape as A containing the softmax-normalized values (computed as exp(A[i] - max(A)) / sum(exp(A[j] - max(A)))) and, if A required gradients, configured to compute gradients through a Softmax gradient function.</returns>
	template <typename T>
	TensorCore::Tensor<T> Softmax(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator); /// Should not be used in actual machine learning contexts, debugging / teaching moment for me only

	/// <summary>
	/// Computes the softmax along the specified axis of the input tensor using a numerically stable implementation (subtracts the maximum before exponentiation). If the input requires gradients, the returned tensor will also require gradients and a corresponding gradient function will be attached. Throws std::out_of_range if axis is out of bounds.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor (e.g., float or double). Must support numeric operations such as std::exp and numeric_limits<T>::infinity().</typeparam>
	/// <param name="A">The input tensor. Softmax will be computed over slices along the specified axis.</param>
	/// <param name="axis">The axis index along which to compute softmax. Must be less than A.Rank(); otherwise std::out_of_range is thrown.</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate memory for the returned tensor.</param>
	/// <returns>A new TensorCore::Tensor<T> with the same shape as A where values along the given axis are replaced by their softmax probabilities (each slice along the axis sums to 1). If A.RequiresGrad() was true, the returned tensor will require gradients and have an attached gradient function.</returns>
	template <typename T>
	TensorCore::Tensor<T> AxisSoftmax(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the log-softmax of the input tensor along the specified axis using a numerically stable method.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor (e.g., float, double) for which the log-softmax is computed.</typeparam>
	/// <param name="A">Input tensor whose log-softmax will be computed. The output has the same shape as this tensor.</param>
	/// <param name="axis">Index of the axis (dimension) along which to compute the log-softmax. Throws std::out_of_range if axis >= A.Rank().</param>
	/// <param name="allocator">Memory allocator used to allocate intermediate and result tensors.</param>
	/// <returns>A tensor of the same shape as A containing the log-softmax values computed along the specified axis.</returns>
	template <typename T>
	TensorCore::Tensor<T> AxisLogSoftmax(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator); /// Because of numerical stability, apparently
}

#include "activation.inl"
