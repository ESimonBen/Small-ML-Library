 /// elementwise.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	/// <summary>
	/// Adds two tensors element-wise, supporting broadcasting. Returns a new tensor with the broadcasted shape.
	/// </summary>
	/// <typeparam name="T">Element type of the input and output tensors.</typeparam>
	/// <param name="A">Left-hand input tensor. Not modified. Must be broadcastable with B.</param>
	/// <param name="B">Right-hand input tensor. Not modified. Must be broadcastable with A.</param>
	/// <param name="allocator">Allocator used to allocate memory for and construct the result tensor.</param>
	/// <returns>A tensor containing the element-wise sum of A and B using broadcast rules. If A or B requires gradients, the result will require gradients and have an appropriate gradient function attached.</returns>
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Add(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Performs element-wise subtraction of two tensors with broadcasting support. Throws std::runtime_error if the tensor shapes cannot be broadcast. If either input requires gradients, the result will require gradients and its gradient function will be set.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensors (e.g., float, double, int).</typeparam>
	/// <param name="A">Const reference to the left-hand tensor operand. Its shape may be equal to B's shape or broadcast-compatible with B.</param>
	/// <param name="B">Const reference to the right-hand tensor operand. Its shape may be equal to A's shape or broadcast-compatible with A.</param>
	/// <param name="allocator">Reference to a Memory::ArenaAllocator used to allocate storage for the result tensor.</param>
	/// <returns>A newly allocated TensorCore::Tensor<T> containing the element-wise difference A - B using the broadcasted shape. If A.RequiresGrad() or B.RequiresGrad() is true, the returned tensor will have RequiresGrad set and an appropriate gradient function attached.</returns>
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Subtract(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Performs elementwise multiplication of two tensors, supporting broadcasting. Throws std::runtime_error if the tensor shapes cannot be broadcast together.
	/// </summary>
	/// <typeparam name="T">Element type of the tensors. Must support multiplication (operator*) and storage in TensorCore::Tensor<T>.</typeparam>
	/// <param name="A">Left-hand input tensor. Not modified. Its values are used (with broadcasting) to compute the product.</param>
	/// <param name="B">Right-hand input tensor. Not modified. Its values are used (with broadcasting) to compute the product.</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate the result tensor's storage.</param>
	/// <returns>A newly allocated TensorCore::Tensor<T> whose shape is the broadcasted shape of A and B and whose elements are the elementwise products. If either input requires gradients, the returned tensor will require gradients and will have its gradient function set to AutoGrad::MulGradFn<T>.</returns>
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Multiply(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Performs element-wise division of two tensors with broadcasting support and returns a newly allocated result tensor. If either input requires gradients, the result is marked to require gradients and a division gradient function is attached. Throws a runtime_error if the tensor shapes cannot be broadcast together.
	/// </summary>
	/// <typeparam name="T">The element type of the tensors (e.g., float, double, int).</typeparam>
	/// <param name="A">The dividend tensor (left operand). May have a different shape than B if broadcastable.</param>
	/// <param name="B">The divisor tensor (right operand). May have a different shape than A if broadcastable.</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate storage for the result tensor.</param>
	/// <returns>A TensorCore::Tensor<T> containing the element-wise quotient of A and B (after applying broadcasting). If A.RequiresGrad() or B.RequiresGrad() is true, the returned tensor will require gradients and have its gradient function set accordingly.</returns>
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Divide(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the element-wise power of a tensor by raising each element of A to the given exponent, producing a new tensor allocated from the provided allocator. If A requires gradients, the returned tensor is configured for automatic differentiation.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor (for example float or double). Must support std::pow(T, T) and any operations required by the tensor implementation.</typeparam>
	/// <param name="A">Input tensor whose elements will be raised to the exponent. Passed by const reference; the result has the same shape as A.</param>
	/// <param name="exponent">Scalar exponent used for the power operation. Must be compatible with std::pow for type T.</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate storage for the resulting tensor.</param>
	/// <returns>A TensorCore::Tensor<T> containing std::pow(A[i], exponent) for each element i. If A.RequiresGrad() is true, the returned tensor will have requiresGrad enabled and an associated gradient function. The function is annotated [[nodiscard]] to encourage using the returned value.</returns>
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Power(const TensorCore::Tensor<T>& A, T exponent, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the element-wise absolute value of the input tensor and returns a new tensor with the same shape, allocated using the provided allocator. If the input requires gradients, the result is marked to require gradients and its gradient function is set accordingly.
	/// </summary>
	/// <typeparam name="T">The element type of the tensor (e.g., float, double, int).</typeparam>
	/// <param name="A">The input tensor (const reference). Its elements are not modified; the returned tensor contains their absolute values.</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate storage for the returned tensor.</param>
	/// <returns>A TensorCore::Tensor<T> with the absolute values of A, having the same shape as A. If A.RequiresGrad() is true, the returned tensor will require gradients and will have its gradient function set to AutoGrad::AbsGradFn<T>.</returns>
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Abs(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Clamps each element of the input tensor A to the range [min, max] and returns a new tensor with the same shape.
	/// </summary>
	/// <typeparam name="T">Element type stored in the tensor.</typeparam>
	/// <param name="A">Input tensor to clamp (const reference).</param>
	/// <param name="min">Inclusive lower bound for clamping.</param>
	/// <param name="max">Inclusive upper bound for clamping.</param>
	/// <param name="allocator">Memory allocator used to allocate the resulting tensor.</param>
	/// <returns>A new Tensor<T> with the same shape as A where each element is clamped to [min, max]. If A.RequiresGrad() is true, the returned tensor will require gradients and will have a ClampGradFn configured. The result is allocated using the provided allocator.</returns>
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Clamp(const TensorCore::Tensor<T>& A, T min, T max, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the element-wise natural logarithm of the input tensor and returns a new tensor with the same shape, allocated using the provided allocator.
	/// </summary>
	/// <typeparam name="T">The numeric element type of the tensor (for example, float or double).</typeparam>
	/// <param name="A">The input tensor whose element-wise natural logarithm will be computed (const reference).</param>
	/// <param name="allocator">Allocator used to allocate storage for the result tensor.</param>
	/// <returns>A tensor containing the element-wise natural logarithm of A. If A.RequiresGrad() is true, the returned tensor will have requiresGrad enabled and a gradient function attached to support backpropagation.</returns>
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Log(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the element-wise exponential of the input tensor.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor. Must be a numeric type compatible with std::exp (typically a floating-point type).</typeparam>
	/// <param name="A">Input tensor (const reference). Each element is passed to std::exp.</param>
	/// <param name="allocator">Arena allocator used to allocate the storage for the returned tensor.</param>
	/// <returns>A TensorCore::Tensor<T> with the same shape as A where each element is std::exp(A[i]). If A.RequiresGrad() is true, the returned tensor will require gradients and its gradient function will be set to AutoGrad::ExpGradFn<T> associated with A's implementation.</returns>
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Exp(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Performs an element-wise equality comparison of two tensors and returns a tensor with 1 where elements are equal and 0 otherwise. Throws std::runtime_error if the input shapes do not match.
	/// </summary>
	/// <typeparam name="T">Element type of the input tensors and of the returned tensor.</typeparam>
	/// <param name="A">The first input tensor. Must have the same shape as B.</param>
	/// <param name="B">The second input tensor. Must have the same shape as A.</param>
	/// <param name="allocator">Allocator used to construct the result tensor.</param>
	/// <returns>A tensor of the same shape as A and B containing static_cast<T>(1) at positions where A and B are equal and static_cast<T>(0) otherwise. The returned tensor has gradients disabled.</returns>
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Equal(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Returns a tensor with each element negated.
	/// </summary>
	/// <typeparam name="T">The element type of the tensor (e.g., float, double, int).</typeparam>
	/// <param name="A">The input tensor whose elements will be negated. If A.RequiresGrad() is true, the result will preserve the requires-grad flag.</param>
	/// <param name="allocator">Allocator used to allocate memory for the resulting tensor.</param>
	/// <returns>A new TensorCore::Tensor<T> containing the element-wise negation of A. The result uses the provided allocator and will have its requires-grad flag set when A.RequiresGrad() is true.</returns>
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Negate(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the element-wise square of a tensor A, allocating the result with the provided allocator and preserving the input's gradient requirement.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor.</typeparam>
	/// <param name="A">The input tensor to square (const reference).</param>
	/// <param name="allocator">Allocator used to allocate memory for the resulting tensor.</param>
	/// <returns>A TensorCore::Tensor<T> containing the element-wise square of A. If A.RequiresGrad() is true, the returned tensor will have its requires-grad flag set.</returns>
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Square(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the element-wise reciprocal of a tensor (1 / A) and returns a new tensor.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor (for example, float or double).</typeparam>
	/// <param name="A">Input tensor whose elements will be inverted (element-wise).</param>
	/// <param name="allocator">Memory allocator used to allocate the resulting tensor.</param>
	/// <returns>A newly allocated TensorCore::Tensor<T> containing the element-wise reciprocals of A. If A.RequiresGrad() is true, the returned tensor will have its requires-grad flag set.</returns>
	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Reciprocal(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator);
}

#include "elementwise.inl"