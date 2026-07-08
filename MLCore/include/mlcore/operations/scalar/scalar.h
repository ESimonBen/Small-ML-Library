 /// scalar.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	/// <summary>
	/// Creates and returns a new tensor in which Scalar has been added element-wise to Input. The output is allocated using the provided allocator. If the input requires gradients, the output will be marked to require gradients and its gradient function will be set accordingly.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor and the scalar value.</typeparam>
	/// <param name="Input">The input tensor whose elements will be incremented by Scalar. The output preserves the input shape. Passed by const reference.</param>
	/// <param name="Scalar">The scalar value to add to each element of the input tensor.</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate storage for the output tensor.</param>
	/// <returns>A TensorCore::Tensor<T> with the same shape as Input containing the element-wise sums. If Input.RequiresGrad() is true, the returned tensor will require gradients and have its gradient function set.</returns>
	template <typename T>
	TensorCore::Tensor<T> AddScalar(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Multiplies each element of a tensor by a scalar and returns a new tensor. If the input tensor requires gradients, the returned tensor is marked to require gradients and a corresponding gradient function is attached.
	/// </summary>
	/// <typeparam name="T">The element type of the tensor (e.g., a numeric type) for both input and output tensors.</typeparam>
	/// <param name="Input">The input tensor whose elements will be multiplied. Its shape is preserved in the output.</param>
	/// <param name="Scalar">The scalar value to multiply each element of the input tensor by.</param>
	/// <param name="allocator">Allocator used to allocate memory for the output tensor.</param>
	/// <returns>A TensorCore::Tensor<T> with the same shape as Input where each element is Input[i] * Scalar. The tensor is allocated with the provided allocator. If Input.RequiresGrad() is true, the returned tensor will require gradients and have the appropriate gradient function attached.</returns>
	template <typename T>
	TensorCore::Tensor<T> MultiplyScalar(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Performs element-wise subtraction between a tensor and a scalar, returning a new tensor. If scalarOnLeft is true each element is computed as (Scalar - element); otherwise as (element - Scalar). If the input tensor requires gradients, the returned tensor will record a gradient function.
	/// </summary>
	/// <typeparam name="T">The element type of the tensor and the scalar (e.g., float, double, int).</typeparam>
	/// <param name="Input">The input tensor whose elements will be involved in the subtraction.</param>
	/// <param name="Scalar">The scalar value to subtract from each tensor element or to subtract each element from, depending on scalarOnLeft.</param>
	/// <param name="allocator">Allocator used to allocate memory for the returned tensor.</param>
	/// <param name="scalarOnLeft">Determines the operand order: true computes Scalar - Input[i]; false computes Input[i] - Scalar.</param>
	/// <returns>A new Tensor<T> containing the result of the element-wise subtraction. If Input.RequiresGrad() is true, the returned tensor will have gradients enabled and an associated backward function.</returns>
	template <typename T>
	TensorCore::Tensor<T> SubtractScalar(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator, bool scalarOnLeft);

	/// <summary>
	/// Performs element-wise division between a tensor and a scalar, returning a new tensor allocated with the given allocator. Throws std::runtime_error on division by zero.
	/// </summary>
	/// <typeparam name="T">The numeric element type of the tensor. Must support division and equality comparisons (used for zero checks) and be compatible with TensorCore::Tensor<T>.</typeparam>
	/// <param name="Input">The input tensor used as the left or right operand depending on scalarOnLeft.</param>
	/// <param name="Scalar">The scalar value used as the other operand. If scalarOnLeft is true it is the dividend (Scalar / Input[i]); otherwise it is the divisor (Input[i] / Scalar).</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate the output tensor's storage.</param>
	/// <param name="scalarOnLeft">If true compute Scalar / Input[i] for each element; if false compute Input[i] / Scalar.</param>
	/// <returns>A new TensorCore::Tensor<T> with the same shape as Input containing the element-wise division results. If Input.RequiresGrad() is true, the returned tensor will require gradients and will have its gradient function set accordingly.</returns>
	template <typename T>
	TensorCore::Tensor<T> DivideScalar(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator, bool scalarOnLeft);
}

#include "scalar.inl"