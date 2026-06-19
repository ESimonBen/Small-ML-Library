 /// activationGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	/// <summary>
	/// Template class that implements the gradient function for the ReLU activation (inherits from GradFn<T>) and performs the backward pass to compute input gradients.
	/// </summary>
	/// <typeparam name="T">Element type of the tensors operated on (e.g., float, double).</typeparam>
	template <typename T>
	class ReLUGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs a ReLUGradFn<T> by initializing the base GradFn<T> with the provided implementation pointer.
		/// </summary>
		/// <typeparam name="T">The element type (e.g., numeric or tensor element type) used by the gradient function.</typeparam>
		/// <param name="a">A shared_ptr to a GradFn<T>::Impl that supplies the underlying implementation; forwarded to the base GradFn<T> constructor.</param>
		ReLUGradFn(std::shared_ptr<typename GradFn<T>::Impl> a);

		/// <summary>
		/// Performs the backward pass for a ReLU activation. Validates shapes, builds the input gradient by masking the output gradient where the input is positive (gradInput[i] = gradOutput[i] if input[i] > 0, otherwise 0), and propagates it by calling input.Backward(gradInput). Throws std::runtime_error if the output gradient and input shapes differ. If the input does not require gradients, the function returns early without allocating or propagating.
		/// </summary>
		/// <typeparam name="T">Element type of the tensors (e.g., float, double, etc.).</typeparam>
		/// <param name="gradOutput">Const reference to the gradient tensor with respect to the ReLU output. Must have the same shape as the stored input tensor.</param>
		/// <param name="allocator">Allocator used to create the gradInput tensor that will hold the gradient with respect to the input.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;
	};

	/// <summary>
	/// Gradient function object for the Leaky ReLU activation. Performs the backward pass to compute gradients with the configured negative-slope parameter (alpha).
	/// </summary>
	/// <typeparam name="T">Numeric type for tensor elements and gradients (for example, float or double).</typeparam>
	template <typename T>
	class LeakyReLUGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructor that creates a LeakyReLUGradFn<T> and initializes it with a previous gradient function implementation and the leaky-ReLU alpha (negative slope).
		/// </summary>
		/// <typeparam name="T">The numeric type used for tensor values and the alpha parameter.</typeparam>
		/// <param name="a">Shared ownership pointer to the underlying GradFn<T>::Impl that this gradient function depends on (previous node/implementation).</param>
		/// <param name="alpha">The leaky-ReLU negative-slope coefficient of type T; stored in the object for use during gradient computation.</param>
		LeakyReLUGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T alpha);

		/// <summary>
		/// Computes the gradient of a LeakyReLU activation w.r.t. its input and propagates it to the previous node.
		/// </summary>
		/// <typeparam name="T">Numeric element type of the tensors (e.g., float, double).</typeparam>
		/// <param name="gradOutput">Incoming gradient tensor (dL/dy) for the activation's output. Must have the same shape as the saved input; otherwise the function throws std::runtime_error.</param>
		/// <param name="allocator">Memory allocator used to allocate the temporary input gradient tensor (dL/dx).</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		T alpha; /// Represents the multiplier (alpha) for elements <= 0
	};

	/// <summary>
	/// Gradient function object for the softmax operation. Performs the backward pass to propagate gradients through a softmax layer.
	/// </summary>
	/// <typeparam name="T">The numeric data type of the elements in the tensors (e.g., float, double).</typeparam>
	template <typename T>
	class SoftmaxGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs a SoftmaxGradFn<T> object, initializing the base GradFn with the provided input implementation and storing the provided output implementation.
		/// </summary>
		/// <typeparam name="T">The element/data type used by the gradient functions (e.g., float, double).</typeparam>
		/// <param name="a">Shared pointer to a GradFn<T>::Impl used to initialize the base gradient function (input implementation).</param>
		/// <param name="b">Shared pointer to a GradFn<T>::Impl that will be stored as the output implementation.</param>
		SoftmaxGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		/// <summary>
		/// Performs the backward pass for a softmax activation, computing and propagating the gradient to the input tensor.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors. Must be a floating-point type.</typeparam>
		/// <param name="gradOutput">Gradient w.r.t. the softmax output. Must have the same shape as the stored input/output tensors.</param>
		/// <param name="allocator">Memory arena allocator used for creating temporary tensors during the backward computation.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		std::shared_ptr<typename GradFn<T>::Impl> outputImpl; /// The output of the operation as a TensorImpl
	};

	/// <summary>
	/// Template gradient function object that computes the gradient of a softmax operation along a specified axis, delegating to two underlying GradFn implementations as needed.
	/// </summary>
	/// <typeparam name="T">The numeric element type of the tensors (e.g., float, double) used by the gradient computations.</typeparam>
	template <typename T>
	class AxisSoftmaxGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs an AxisSoftmaxGradFn that computes the softmax gradient along a specified axis.
		/// </summary>
		/// <typeparam name="T">The numeric type used for gradients and internal computations (e.g., float or double).</typeparam>
		/// <param name="a">Shared pointer to a GradFn<T>::Impl used to initialize the base GradFn (previous/input gradient function).</param>
		/// <param name="b">Shared pointer to a GradFn<T>::Impl stored as the output implementation (output gradient function).</param>
		/// <param name="axis">The axis (dimension) along which the softmax gradient is computed.</param>
		AxisSoftmaxGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b, size_t axis);

		/// <summary>
		/// Computes and propagates the gradient of a softmax operation along the configured axis. If the input does not require gradients, the function returns early.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors (e.g., float, double).</typeparam>
		/// <param name="gradOutput">Gradient of the loss with respect to the softmax output (dL/dy). Provided as a tensor with the same shape as the output.</param>
		/// <param name="allocator">Arena allocator used to allocate the tensor for the computed input gradients.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		std::shared_ptr<typename GradFn<T>::Impl> outputImpl; /// The output of the operation as a TensorImpl
		size_t axis; /// Represents the axis that was chosen to Softmax
	};
}

#include "activationGradFn.inl"