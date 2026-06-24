 /// reduction.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	/// <summary>
	/// Returns a tensor containing the sum of all elements in the input tensor.
	/// </summary>
	/// <typeparam name="T">The element type of the tensor. Must be an arithmetic type (compile-time checked via static_assert).</typeparam>
	/// <param name="A">The input tensor whose elements will be summed. Passed by const reference.</param>
	/// <param name="allocator">The memory arena allocator used to construct and store the returned single-element tensor.</param>
	/// <returns>A TensorCore::Tensor<T> with shape {1} whose sole element is the sum of all elements in A (or zero if A is empty). If A.RequiresGrad() is true, the returned tensor will require gradients and have its gradient function set appropriately.</returns>
	template <typename T>
	TensorCore::Tensor<T> SumAll(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the mean of all elements in a tensor and returns it as a scalar tensor. The template type T must be a floating-point type.
	/// </summary>
	/// <typeparam name="T">The floating-point element type of the tensor (required to be a floating-point type).</typeparam>
	/// <param name="A">The input tensor whose elements will be averaged. Must contain at least one element; if empty, the function throws a runtime_error. If A.RequiresGrad() is true, the result will inherit the requires-grad flag.</param>
	/// <param name="allocator">An ArenaAllocator used for any internal allocations and for allocating the result tensor.</param>
	/// <returns>A scalar Tensor<T> containing the mean (sum of all elements divided by the number of elements).</returns>
	template <typename T>
	TensorCore::Tensor<T> MeanAll(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the maximum element of the input tensor and returns it as a scalar (1-element) tensor. The function scans all elements (O(n)). Throws std::runtime_error if the input tensor is empty. T is required to satisfy std::totally_ordered.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensor. Must satisfy std::totally_ordered (i.e., support ordering comparisons).</typeparam>
	/// <param name="A">The input tensor whose maximum element will be computed. Must be non-empty. If A.RequiresGrad() is true, the returned tensor will be configured for gradient propagation.</param>
	/// <param name="allocator">Allocator used to construct and own the returned scalar tensor.</param>
	/// <returns>A TensorCore::Tensor<T> with shape {1} containing the maximum element of A. If A.RequiresGrad() is true, the returned tensor is marked to require gradients and its gradient function is set accordingly. Throws std::runtime_error if A has no elements.</returns>
	template <typename T>
	TensorCore::Tensor<T> MaxAll(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the minimum value over all elements of the input tensor and returns it as a scalar tensor. Requires that T is totally ordered and throws at runtime if the input tensor is empty.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor. Must satisfy std::totally_ordered (compile-time requirement).</typeparam>
	/// <param name="A">Input tensor to reduce. Must contain at least one element; if empty, the function throws std::runtime_error. If A.RequiresGrad() is true, the returned tensor will be marked to require gradients and a gradient function will be attached.</param>
	/// <param name="allocator">Arena allocator used to construct and allocate the resulting scalar tensor.</param>
	/// <returns>A TensorCore::Tensor<T> of shape {1} holding the minimum element of A. The returned tensor may require gradients and will have its gradient function set when A.RequiresGrad() is true.</returns>
	template <typename T>
	TensorCore::Tensor<T> MinAll(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the sum of tensor elements along the specified axis and returns a new tensor with that axis reduced. If keepDims is true, the reduced axis is kept with size 1; otherwise it is removed. Throws std::out_of_range if the axis is out of bounds. If the input tensor requires gradients, the result is configured for automatic differentiation.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor (for example float, double, int). Must be default-constructible and support zero-initialization and addition (i.e., convertible from static_cast<T>(0) and support operator+=/operator+).</typeparam>
	/// <param name="A">Input tensor whose elements will be summed along the given axis. Its rank and dimensions determine the valid axis and the output shape. If A.RequiresGrad() is true, a gradient function will be attached to the result.</param>
	/// <param name="axis">Zero-based index of the axis to sum over. Must be less than A.Rank(); otherwise std::out_of_range is thrown.</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate storage for the output tensor.</param>
	/// <param name="keepDims">If true, the reduced axis is retained in the result with size 1. If false, the axis is removed; if all axes are removed, the function returns a tensor with a single dimension of size 1.</param>
	/// <returns>A TensorCore::Tensor<T> containing the sums over the specified axis. The shape reflects the keepDims behavior. If the input required gradients, the returned tensor will have requiresGrad set and an associated gradient function.</returns>
	template <typename T>
	TensorCore::Tensor<T> AxisSum(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator, bool keepDims = false);

	/// <summary>
	/// Computes the mean of tensor A along the specified axis by summing over that axis and dividing by its size. Throws std::out_of_range if the axis is out of bounds.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor (numeric type that supports summation and division).</typeparam>
	/// <param name="A">Input tensor whose elements will be averaged along the given axis.</param>
	/// <param name="axis">Index of the dimension to reduce (must be less than A.Rank()).</param>
	/// <param name="allocator">Memory allocator used for creating the result and any intermediate tensors.</param>
	/// <param name="keepDims">If true, the reduced dimension is kept with size 1; if false, the rank is reduced by one.</param>
	/// <returns>A tensor containing the mean values computed along the specified axis. Shape reflects keepDims: same rank with size 1 along the axis when true, or reduced rank when false.</returns>
	template <typename T>
	TensorCore::Tensor<T> AxisMean(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator, bool keepDims = false);

	/// <summary>
	/// Computes the maximum values of tensor A along the specified axis and returns a new tensor containing those maxima. If keepDims is true the reduced axis is retained with size 1; otherwise the axis is removed. If axis is out of range the function throws std::out_of_range. If the input requires gradients, the returned tensor is configured for autograd with an AxisMax gradient function.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor. Must support comparison and std::numeric_limits<T>::infinity().</typeparam>
	/// <param name="A">Const reference to the input tensor whose values are reduced.</param>
	/// <param name="axis">Index of the axis to reduce (must be less than A.Rank()).</param>
	/// <param name="allocator">Allocator used to allocate memory for the result tensor.</param>
	/// <param name="keepDims">If true, keep the reduced axis as a dimension of size 1; if false, remove the axis (if all axes are removed the result becomes a single-element tensor represented with a 1-D shape of length 1).</param>
	/// <returns>A Tensor<T> containing the maximum values along the given axis. The shape reflects keepDims behavior. If A.RequiresGrad() is true, the returned tensor will have requiresGrad set and a gradient function attached to propagate gradients.</returns>
	template <typename T>
	TensorCore::Tensor<T> AxisMax(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator, bool keepDims = false);

	/// <summary>
	/// Computes the minimum values of a tensor along the specified axis.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor. Must support comparison operations and numeric_limits<T>::infinity().</typeparam>
	/// <param name="A">The input tensor to reduce. Values are compared along the specified axis.</param>
	/// <param name="axis">Index of the axis to reduce (must be less than A.Rank()).</param>
	/// <param name="allocator">Allocator used to construct and store the resulting tensor.</param>
	/// <param name="keepDims">If true, the reduced axis is retained with size 1; otherwise the axis is removed (result will be a scalar tensor with dimension 1 if all axes are reduced).</param>
	/// <returns>A TensorCore::Tensor<T> containing the minimum values computed along the given axis. The output tensor shape respects the keepDims flag. Throws std::out_of_range if the axis is out of bounds. If the input requires gradients, the returned tensor will be marked to require gradients and will have an associated gradient function.</returns>
	template <typename T>
	TensorCore::Tensor<T> AxisMin(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator, bool keepDims = false);
}

#include "reduction.inl"
