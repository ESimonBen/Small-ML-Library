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

	/// <summary>
	/// Template gradient function that reshapes or reduces a gradient tensor back to a specified original shape during the backward pass.
	/// </summary>
	/// <typeparam name="T">The element type of the tensors (numeric type) handled by this gradient function.</typeparam>
	template <typename T>
	class ReduceToShapeGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs a ReduceToShapeGradFn<T>, forwarding the provided gradient implementation to the base GradFn and storing the original shape.
		/// </summary>
		/// <typeparam name="T">The value/gradient element type handled by this gradient function.</typeparam>
		/// <param name="a">Shared pointer to the underlying GradFn<T>::Impl that provides the gradient implementation; forwarded to the base GradFn<T> constructor.</param>
		/// <param name="originalShape">Reference to a Utils::Shape describing the original/target shape; its value is copied into the member m_OriginalShape.</param>
		ReduceToShapeGradFn(std::shared_ptr<typename GradFn<T>::Impl> a);

		/// <summary>
		/// Propagates the gradient for a reduce-to-shape operation by expanding the output gradient back to the original input shape and calling the input's backward.
		/// </summary>
		/// <typeparam name="T">Element type of the tensors (for example, float or double).</typeparam>
		/// <param name="gradOutput">Gradient tensor with respect to this function's output. It is detached and then expanded to the input's original shape to form the input gradient.</param>
		/// <param name="allocator">Allocator used to allocate memory when expanding the detached output gradient to the input's original shape.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		Utils::Shape m_OriginalShape; /// The shape of the input tensor
	};

	/// <summary>
	/// Gradient function object that computes the backward pass for tensors that were expanded to a larger shape. It maps gradients from the expanded shape back to the original shape during backpropagation.
	/// </summary>
	/// <typeparam name="T">Element type of the tensors handled by this gradient function (e.g., float, double, int).</typeparam>
	template <typename T>
	class ExpandToShapeGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs an ExpandToShapeGradFn that wraps a child gradient function and records the original shape to which gradients should be expanded.
		/// </summary>
		/// <typeparam name="T">The element type for which gradients are computed.</typeparam>
		/// <param name="a">A shared pointer to the child gradient function implementation (GradFn<T>::Impl). This is forwarded to the base GradFn<T> constructor.</param>
		/// <param name="originalShape">A reference to the original tensor shape. It is used to initialize the member m_OriginalShape (copied into the object) and determines the target shape for expanding gradients.</param>
		ExpandToShapeGradFn(std::shared_ptr<typename GradFn<T>::Impl> a);

		/// <summary>
		/// Performs backward gradient propagation for an expand-to-shape operation: if the original input requires gradients, detaches the output gradient, reduces it to the input's original shape, and calls the input's Backward with the reduced gradient. Does nothing if the input does not require gradients.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors (for example float or double).</typeparam>
		/// <param name="gradOutput">The gradient tensor with respect to the operation's output. It is detached and used as the source for reduction back to the input shape.</param>
		/// <param name="allocator">Memory arena allocator used for temporary allocations during the reduction (e.g., by ReduceSumToShape).</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		Utils::Shape m_OriginalShape; /// The shape of the input tensor
	};
}

#include "broadcastGradFn.inl"