 /// scalarGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	/// <summary>
	/// Gradient function object for the addition of a scalar to a tensor; implements the backward pass for the add-scalar operation.
	/// </summary>
	/// <typeparam name="T">The element type of the tensors (e.g., float, double) used by the forward and backward computations.</typeparam>
	template <typename T>
	class AddScalarGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs an AddScalarGradFn by forwarding the provided implementation pointer to the base GradFn<T> constructor.
		/// </summary>
		/// <typeparam name="T">The value/gradient type associated with this gradient function.</typeparam>
		/// <param name="a">A shared pointer to a GradFn<T>::Impl representing the underlying gradient-function implementation to wrap.</param>
		AddScalarGradFn(std::shared_ptr<typename GradFn<T>::Impl> a);

		/// <summary>
		/// Performs the backward pass for an AddScalar operation by propagating the output gradient to the input if the input requires gradients.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors (e.g., float, double, int).</typeparam>
		/// <param name="gradOutput">The gradient tensor received from the next operation (should match the input shape). This gradient is forwarded to the input.</param>
		/// <param name="allocator">An arena allocator available for any temporary memory allocations needed during the backward computation. The implementation may not always allocate memory.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;
	};

	/// <summary>
	/// Gradient function object for a tensor-scalar subtraction operation. Implements GradFn<T> and performs the backward pass for cases where one operand is a scalar.
	/// </summary>
	/// <typeparam name="T">Element type of the tensors (e.g., float, double).</typeparam>
	template <typename T>
	class SubScalarGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Initializes a SubScalarGradFn<T> by wrapping the provided gradient implementation and recording whether the scalar operand is on the left.
		/// </summary>
		/// <typeparam name="T">The value type used by the gradient function implementation.</typeparam>
		/// <param name="a">Shared pointer to the underlying gradient function implementation (GradFn<T>::Impl) that this object wraps.</param>
		/// <param name="scalarOnLeft">If true, indicates the scalar operand is on the left side of the subtraction; if false, the scalar is on the right.</param>
		SubScalarGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, bool scalarOnLeft);

		/// <summary>
		/// Performs the backward pass for a tensor minus scalar (sub-scalar) operation, computing and propagating the appropriate gradient to the input tensor.
		/// </summary>
		/// <typeparam name="T">The element/data type of the tensors (for example, float or double).</typeparam>
		/// <param name="gradOutput">The gradient tensor received from subsequent operations (detached before use).</param>
		/// <param name="allocator">Arena allocator used to allocate any temporary tensors needed during the backward computation.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		bool scalarOnLeft; /// Flag indicating whether a scalar operand is positioned on the left side of an operation.
	};

	/// <summary>
	/// Gradient function node that scales incoming gradients by a given scalar during the backward pass.
	/// </summary>
	/// <typeparam name="T">The numeric element type used by tensors and the scalar (e.g., float, double).</typeparam>
	template <typename T>
	class MulScalarGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs a MulScalarGradFn<T> that wraps a gradient implementation and stores a scalar multiplier.
		/// </summary>
		/// <typeparam name="T">The numeric type used for gradient values and the scalar multiplier.</typeparam>
		/// <param name="a">Shared pointer to the underlying GradFn<T>::Impl that provides the gradient implementation.</param>
		/// <param name="scalar">Scalar value of type T to be used for multiplying the gradient; stored in the instance.</param>
		MulScalarGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T scalar);

		/// <summary>
		/// Backpropagates the gradient through a multiply-by-scalar operation: if the input requires gradients, multiplies the outgoing gradient by the stored scalar and passes the resulting gradient to the input's backward method.
		/// </summary>
		/// <typeparam name="T">The numeric data type of the tensor elements (e.g., float, double) used by the input, output, and gradient tensors.</typeparam>
		/// <param name="gradOutput">The gradient tensor from the subsequent operation (w.r.t. this node's output). It is detached and used as the source gradient to be scaled and propagated backward.</param>
		/// <param name="allocator">Memory arena allocator used to allocate any intermediate tensors (for example the scaled gradient) during backward computation.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		T scalar;
	};

	/// <summary>
	/// Gradient function object that performs the backward pass for a division-by-scalar operation. It holds the scalar and a flag indicating whether the scalar was on the left (scalar / tensor) or on the right (tensor / scalar) and delegates gradient propagation to the underlying GradFn implementation.
	/// </summary>
	/// <typeparam name="T">Element type for tensors and the scalar (e.g., float or double).</typeparam>
	template <typename T>
	class DivScalarGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs a DivScalarGradFn<T> that represents division between a gradient function and a scalar.
		/// </summary>
		/// <typeparam name="T">The numeric type of the values handled by the gradient function (e.g., float or double).</typeparam>
		/// <param name="a">Shared pointer to the underlying GradFn<T>::Impl to wrap and compute gradients from.</param>
		/// <param name="scalar">The scalar value used in the division operation.</param>
		/// <param name="scalarOnLeft">If true, the operation is scalar / a; if false, the operation is a / scalar.</param>
		DivScalarGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T scalar, bool scalarOnLeft);

		/// <summary>
		/// Performs the backward pass for a division-by-scalar operation. Computes the gradient with respect to the input tensor (taking into account whether the scalar was on the left or right) and calls input.Backward to propagate it. The function detaches inputs/gradients to avoid creating new graph edges and uses the provided allocator for intermediate tensors. If the input does not require gradients, it returns immediately.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensors (for example float or double).</typeparam>
		/// <param name="gradOutput">The gradient tensor with respect to the operation's output. This is detached inside the function and used to compute the input gradient.</param>
		/// <param name="allocator">Memory::ArenaAllocator used to allocate intermediate tensors (e.g., squares, divisions, multiplications) during the backward computation.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		T scalar; /// Scalar of type T that the input tensor was divided by
		bool scalarOnLeft; /// Flag indicating whether a scalar operand is positioned on the left side of an operation.
	};
}

#include "scalarGradFn.inl"