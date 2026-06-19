 /// broadcastGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	/// <summary>
	/// Gradient function object for the squeeze operation that performs the backward pass for tensors with element type T.
	/// </summary>
	/// <typeparam name="T">The element type of the tensors handled by this gradient function (for example, float or double).</typeparam>
	template <typename T>
	class SqueezeGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs a SqueezeGradFn<T> that initializes gradient computation for a squeeze operation along a given axis.
		/// </summary>
		/// <typeparam name="T">The data type (e.g., tensor element type) used by the gradient function.</typeparam>
		/// <param name="a">A shared pointer to the underlying GradFn<T>::Impl, representing the previous or associated gradient function implementation.</param>
		/// <param name="axis">The axis along which the original squeeze operation was applied; used to compute the corresponding gradient behavior.</param>
		SqueezeGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis);

		/// <summary>
		/// Performs the backward pass for a squeeze operation: if the input requires gradients, it unsqueezes the output gradient along the stored axis to produce the input gradient and propagates it to the input tensor.
		/// </summary>
		/// <typeparam name="T">Element type of the tensors (for example, float or double).</typeparam>
		/// <param name="gradOutput">Const reference to the gradient tensor with respect to the operation's output; used to compute the gradient for the input.</param>
		/// <param name="allocator">Reference to an arena allocator used to allocate temporary tensors (e.g., for the unsqueezed gradient).</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		size_t m_Axis; /// Represents the axis that was squeezed.
	};

	/// <summary>
	/// Gradient function object for the unsqueeze operation. On the backward pass it removes the singleton dimension that was added at the specified axis so the gradient shape matches the original input.
	/// </summary>
	/// <typeparam name="T">The element type of the tensors (e.g., float, double, int).</typeparam>
	template <typename T>
	class UnsqueezeGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructor that creates an UnsqueezeGradFn instance by attaching a preceding gradient function and recording the axis used for the unsqueeze operation.
		/// </summary>
		/// <typeparam name="T">The value type used by the gradient functions (the template parameter for GradFn and UnsqueezeGradFn).</typeparam>
		/// <param name="a">Shared pointer to the implementation of the preceding gradient function (GradFn<T>::Impl) that this node will wrap.</param>
		/// <param name="axis">The axis index where the unsqueeze was applied; stored in m_Axis.</param>
		UnsqueezeGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis);

		/// <summary>
		/// Performs the backward pass for an Unsqueeze operation: computes the gradient for the input by squeezing the output gradient along the stored axis and then propagates it to the input tensor.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors (e.g., float, double, int).</typeparam>
		/// <param name="gradOutput">The gradient with respect to the operation's output. This tensor is detached and used to compute the input gradient by applying a squeeze along the member axis.</param>
		/// <param name="allocator">Arena allocator used to allocate any temporary tensors required during the squeeze operation.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		size_t m_Axis; /// Represents the axis that was unsqueezed.
	};
}

#include "broadcastGradFn.inl"