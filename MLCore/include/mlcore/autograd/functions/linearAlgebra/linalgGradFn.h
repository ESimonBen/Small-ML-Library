 /// linalgGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/linearAlgebra/linalg.h>

namespace MLCore::AutoGrad {
	/// <summary>
	/// Gradient function object for a dot-product operation. Constructs a DotGradFn that stores gradient implementations for the two operands and provides a Backward override to propagate output gradients to those operands.
	/// </summary>
	/// <typeparam name="T">The element type of the tensors and gradients (for example, float or double).</typeparam>
	template <typename T>
	class DotGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs a DotGradFn<T> by initializing the base GradFn<T> with two gradient function implementations.
		/// </summary>
		/// <typeparam name="T">The numeric or tensor element type used by the gradient functions.</typeparam>
		/// <param name="a">A shared pointer to the first GradFn<T>::Impl used by this DotGradFn.</param>
		/// <param name="b">A shared pointer to the second GradFn<T>::Impl used by this DotGradFn.</param>
		DotGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		/// <summary>
		/// Backpropagates a scalar gradient through a dot-product operation: validates shapes, extracts the scalar gradient, and, if needed, computes and dispatches per-input gradients by multiplying the detached other input by the scalar and calling their Backward methods.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors (numeric type used for values and gradient calculations).</typeparam>
		/// <param name="gradOutput">A tensor containing the gradient w.r.t. the output. Must be a scalar tensor (NumElements() == 1). If its shape is invalid, the function throws a runtime_error.</param>
		/// <param name="allocator">Memory allocator used to allocate intermediate gradient tensors (e.g., for MultiplyScalar outputs).</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;
	};

	/// <summary>
	/// Gradient function object for a matrix multiplication operation. It holds implementations for the input operands and performs the backward pass to compute and propagate input gradients.
	/// </summary>
	/// <typeparam name="T">The element type of the tensors (e.g., float, double) used by the matmul and its gradients.</typeparam>
	template <typename T>
	class MatMulGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs a MatMulGradFn<T> and initializes its base GradFn<T> with two gradient-function implementations for the operands of a matrix multiplication.
		/// </summary>
		/// <typeparam name="T">The numeric type used by the gradient functions (element type for the matrices).</typeparam>
		/// <param name="a">A shared pointer to the gradient-function implementation for the first (left) operand.</param>
		/// <param name="b">A shared pointer to the gradient-function implementation for the second (right) operand.</param>
		MatMulGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		/// <summary>
		/// Backpropagates gradients for a matrix multiplication node by computing and passing gradients to input tensors.
		/// </summary>
		/// <typeparam name="T">Element type of the tensors (e.g., float, double).</typeparam>
		/// <param name="gradOutput">The gradient tensor with respect to the output of the matrix multiplication. Its shape must match the output shape (rows of the first input, columns of the second input).</param>
		/// <param name="allocator">Allocator used for intermediate memory allocations during transpose and matrix multiplication operations.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;
	};

	/// <summary>
	/// Gradient function that computes and applies the gradient for a transpose operation; implements the GradFn<T> interface.
	/// </summary>
	/// <typeparam name="T">Element type of the tensors processed by this gradient function.</typeparam>
	template <typename T>
	class TransposeGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Initializes a TransposeGradFn<T> by forwarding the given GradFn implementation to the base GradFn constructor.
		/// </summary>
		/// <typeparam name="T">The type associated with the gradient function (e.g., element or tensor type).</typeparam>
		/// <param name="a">A shared pointer to a GradFn<T>::Impl that provides the implementation used to initialize the base GradFn.</param>
		TransposeGradFn(std::shared_ptr<typename GradFn<T>::Impl> a);

		/// <summary>
		/// Performs the backward pass for a transpose operation: validates shapes, transposes the incoming gradient, and propagates it to the input if the input requires gradients.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors (e.g., float, double, int).</typeparam>
		/// <param name="gradOutput">The gradient tensor with respect to the output of the transpose operation. Expected shape: gradOutput.Dims()[0] == input.Dims()[1] and gradOutput.Dims()[1] == input.Dims()[0].</param>
		/// <param name="allocator">Arena allocator used to allocate memory for intermediate tensors (for example, the transposed gradient).</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;
	};
}

#include "linalgGradFn.inl"