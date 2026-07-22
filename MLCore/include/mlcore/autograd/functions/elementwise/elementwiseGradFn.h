 /// elementwiseGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	/// <summary>
	/// Gradient function object that combines (adds) gradients from two child gradient implementations during backpropagation.
	/// </summary>
	/// <typeparam name="T">The element type of tensors and gradients (for example, float or double).</typeparam>
	template <typename T>
	class AddGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs an AddGradFn<T> that combines two GradFn<T>::Impl instances.
		/// </summary>
		/// <typeparam name="T">The value type used by GradFn and its implementation (the element type for gradients).</typeparam>
		/// <param name="a">Shared pointer to the first GradFn<T>::Impl to include in the combined function.</param>
		/// <param name="b">Shared pointer to the second GradFn<T>::Impl to include in the combined function.</param>
		AddGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		/// <summary>
		/// Performs the backward pass for an elementwise addition node: propagates the output gradient to each input, reducing the gradient to the input shapes as needed, and calls each input's Backward if it requires gradients.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors (e.g., float, double, etc.).</typeparam>
		/// <param name="gradOutput">The gradient tensor with respect to the output of the addition; will be reduced to match each input's shape as needed.</param>
		/// <param name="allocator">Arena allocator used for any temporary memory allocations during gradient computation.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;
	};

	/// <summary>
	/// Gradient function node that represents the subtraction operation between two gradient implementations and performs the backward pass for that operation.
	/// </summary>
	/// <typeparam name="T">The element type of the tensors (for example float or double) used by the gradient functions and tensors.</typeparam>
	template <typename T>
	class SubGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs a SubGradFn<T> from two gradient implementation instances, delegating initialization to the base GradFn<T>.
		/// </summary>
		/// <typeparam name="T">The element/value type used by the gradient functions.</typeparam>
		/// <param name="a">Shared pointer to the first GradFn<T>::Impl instance to include.</param>
		/// <param name="b">Shared pointer to the second GradFn<T>::Impl instance to include.</param>
		SubGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		/// <summary>
		/// Performs the backward pass for a subtraction operation: distributes gradOutput to the two input tensors (adds reduced gradient to the first input and the negated, reduced gradient to the second) and calls Backward on inputs that require gradients. Throws std::runtime_error if either input is null.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors (e.g., float, double) handled by this gradient function.</typeparam>
		/// <param name="gradOutput">The gradient tensor propagated from subsequent layers (const reference). This gradient is detached and then reduced or negated as required before being applied to inputs.</param>
		/// <param name="allocator">Arena allocator used for temporary allocations during gradient operations (e.g., for reductions or negation).</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator);
	};

	/// <summary>
	/// Gradient function for a multiplication operation that implements backward propagation of gradients to its operands.
	/// </summary>
	/// <typeparam name="T">The numeric element type of the tensors and computations (for example float or double).</typeparam>
	template <typename T>
	class MulGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs a MulGradFn<T> initialized with two gradient function implementations.
		/// </summary>
		/// <typeparam name="T">The value type handled by the gradient functions.</typeparam>
		/// <param name="a">Shared pointer to the first GradFn<T>::Impl, representing the left operand's gradient implementation.</param>
		/// <param name="b">Shared pointer to the second GradFn<T>::Impl, representing the right operand's gradient implementation.</param>
		MulGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		/// <summary>
		/// Performs the backward pass for an element-wise multiplication node: computes and propagates gradients to its input tensors. If an input requires gradients, it multiplies the output gradient by the other input (detached), reduces the result to the input's shape, and calls Backward on that input. Throws std::runtime_error if an input is null.
		/// </summary>
		/// <typeparam name="T">The scalar element type of the tensors (e.g., float, double, int).</typeparam>
		/// <param name="gradOutput">The gradient tensor with respect to this node's output. Used to compute input gradients and should match the output shape.</param>
		/// <param name="allocator">Allocator used for temporary memory during gradient computations (passed to operations such as Multiply).</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;
	};

	/// <summary>
	/// Gradient node that computes the backward pass for a division operation. Inherits from GradFn<T> and implements Backward to produce gradients for its operands.
	/// </summary>
	/// <typeparam name="T">Element type of the tensors (for example float or double).</typeparam>
	template <typename T>
	class DivGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs a DivGradFn<T> by initializing its base GradFn<T> with two implementation instances.
		/// </summary>
		/// <typeparam name="T">The value type used by GradFn and its implementation type (GradFn<T>::Impl).</typeparam>
		/// <param name="a">A shared pointer to a GradFn<T>::Impl instance to be passed to the base GradFn<T>.</param>
		/// <param name="b">A shared pointer to a second GradFn<T>::Impl instance to be passed to the base GradFn<T>.</param>
		DivGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		/// <summary>
		/// Performs the backward pass for a division operation: validates inputs, detaches tensors as needed, computes gradients for each input, and calls Backward on inputs that require gradients. For input a, computes gradA = Operations::ReduceSumToShape(gradOutput / b, a.shape). For input b, computes gradB = Operations::ReduceSumToShape(-gradOutput * (a / b^2), b.shape).
		/// </summary>
		/// <typeparam name="T">The element type of the tensors (e.g., float, double).</typeparam>
		/// <param name="gradOutput">The gradient of the loss with respect to the output of the division (const Tensor<T>&). Used to compute gradients for the inputs.</param>
		/// <param name="allocator">Memory::ArenaAllocator used to allocate intermediate tensors during gradient computation.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;
	};

	/// <summary>
	/// Gradient function object that implements the backward pass for a power operation (raising inputs to a specified exponent).
	/// </summary>
	/// <typeparam name="T">Numeric type of the tensor elements (e.g., float or double).</typeparam>
	template <typename T>
	class PowerGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Initializes a PowerGradFn that wraps an existing gradient-function implementation and applies a power operation with the specified exponent.
		/// </summary>
		/// <typeparam name="T">The value type used by the gradient function (e.g., float, double or a user-defined numeric type).</typeparam>
		/// <param name="a">A shared_ptr to the underlying implementation of the operand gradient function (GradFn<T>::Impl). Ownership is shared with other holders of the pointer.</param>
		/// <param name="exponent">The exponent value to apply to the operand; stored in the object's exponent member.</param>
		PowerGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T exponent);

		/// <summary>
		/// Computes and propagates the gradient for a power operation. It builds the input gradient as gradInput = gradOutput * exponent * base^(exponent - 1) and calls input.Backward(gradInput). If the input does not require gradients, the method returns without action.
		/// </summary>
		/// <typeparam name="T">The numeric element type of the tensors (e.g., float or double).</typeparam>
		/// <param name="gradOutput">The gradient tensor with respect to the output of the power operation (const reference). Used to compute the gradient with respect to the input.</param>
		/// <param name="allocator">Memory::ArenaAllocator used to allocate intermediate tensors during gradient computation.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		T exponent; /// Exponent of the input tensor of type T.
	};

	/// <summary>
	/// Gradient function that computes the backward pass for the element-wise absolute value operation.
	/// </summary>
	/// <typeparam name="T">The element type of the tensors (for example float or double).</typeparam>
	template <typename T>
	class AbsGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs an AbsGradFn<T> by initializing its base GradFn<T> with the provided implementation pointer.
		/// </summary>
		/// <typeparam name="T">The value type used by the gradient function.</typeparam>
		/// <param name="input">A shared_ptr to a GradFn<T>::Impl that represents the underlying function implementation; it is forwarded to the base class constructor.</param>
		AbsGradFn(std::shared_ptr<typename GradFn<T>::Impl> input);

		/// <summary>
		/// Computes and propagates the gradient for an element-wise absolute value operation. If the stored input is null, throws a runtime_error. If the input does not require gradients, returns immediately. Otherwise allocates a gradInput tensor using the provided allocator, sets gradInput[i] = gradOutput[i] * sign(input[i]) where sign is -1 for negative, 1 for positive, and 0 for zero, then calls input.Backward(gradInput).
		/// </summary>
		/// <typeparam name="T">Element type of the tensors (numeric type). Must support comparison with zero and multiplication.</typeparam>
		/// <param name="gradOutput">Constant reference to the gradient of the loss with respect to the output of the absolute-value operation; expected to have the same shape as the input/output.</param>
		/// <param name="allocator">Allocator used to construct the gradInput tensor that will be passed to the input's backward method.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;
	};

	/// <summary>
	/// Gradient function for the clamp operation that computes and propagates gradients for inputs clamped to the range [min, max] during the backward pass.
	/// </summary>
	/// <typeparam name="T">Element type for tensors and scalar bounds (e.g., float, double, int).</typeparam>
	template <typename T>
	class ClampGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Initializes a ClampGradFn that clamps gradient values to the specified range.
		/// </summary>
		/// <typeparam name="T">Type of the gradient values.</typeparam>
		/// <param name="input">Shared pointer to the input gradient function implementation used as the source of gradients.</param>
		/// <param name="min">Lower bound for clamping; values below this are raised to this value.</param>
		/// <param name="max">Upper bound for clamping; values above this are lowered to this value.</param>
		ClampGradFn(std::shared_ptr<typename GradFn<T>::Impl> input, T min, T max);

		/// <summary>
		/// Performs the backward pass for a clamp operation: computes gradients for the input by copying gradOutput where the input value is strictly between m_Min and m_Max, zeroing gradients elsewhere, and then propagates the resulting gradInput to the input tensor. Throws std::runtime_error if the stored input is null and returns early if the input does not require gradients.
		/// </summary>
		/// <typeparam name="T">Element type of the tensors (e.g., float, double, int) used for values and gradients.</typeparam>
		/// <param name="gradOutput">Const reference to the gradient tensor for the output. Must have the same shape as the original input; its values are propagated to gradInput where the input values are in (m_Min, m_Max).</param>
		/// <param name="allocator">Reference to an ArenaAllocator used to allocate the temporary gradInput tensor storage.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		T m_Min;  /// Minimum value to clamp to
		T m_Max;  /// Maximum value to clamp to
	};

	/// <summary>
	/// Gradient function for the logarithm operation that performs the backward pass to propagate gradients.
	/// </summary>
	/// <typeparam name="T">The numeric data type of tensor elements (for example, float or double).</typeparam>
	template <typename T>
	class LogGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs a LogGradFn<T> by initializing its base GradFn<T> with the provided implementation pointer.
		/// </summary>
		/// <typeparam name="T">The numeric or data type used by the gradient function.</typeparam>
		/// <param name="input">A shared_ptr to GradFn<T>::Impl that supplies the underlying implementation or input for this LogGradFn.</param>
		LogGradFn(std::shared_ptr<typename GradFn<T>::Impl> input);

		/// <summary>
		/// Performs the backward pass for a logarithm operation: computes the gradient with respect to the input and propagates it downstream.
		/// </summary>
		/// <typeparam name="T">Element type of the tensors (e.g., float, double).</typeparam>
		/// <param name="gradOutput">Constant reference to the gradient tensor with respect to the output; used to compute the input gradient.</param>
		/// <param name="allocator">Arena allocator used to allocate temporary tensors during the gradient computation.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;
	};

	/// <summary>
	/// Gradient function node that computes and applies the derivative for the exponential operation during backpropagation.
	/// </summary>
	/// <typeparam name="T">The numeric element type used by the tensors (e.g., float, double).</typeparam>
	template <typename T>
	class ExpGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs an ExpGradFn<T> by initializing the base GradFn<T> with the provided implementation pointer.
		/// </summary>
		/// <typeparam name="T">The value type used by the gradient function and its implementation.</typeparam>
		/// <param name="input">A shared pointer to a GradFn<T>::Impl instance that provides the underlying implementation; forwarded to initialize the base GradFn<T>.</param>
		ExpGradFn(std::shared_ptr<typename GradFn<T>::Impl> input);

		/// <summary>
		/// Performs the backward pass for an exponential operation: computes exp(input) element-wise, multiplies it by gradOutput to form the gradient for the input, and propagates that gradient to the input tensor. If the stored input is null a std::runtime_error is thrown; if the input does not require gradients the function returns immediately.
		/// </summary>
		/// <typeparam name="T">The element/data type of the tensors (e.g., float, double, etc.).</typeparam>
		/// <param name="gradOutput">The gradient of the loss with respect to this node's output (const reference). It is detached and multiplied element-wise by exp(input) to produce the input gradient.</param>
		/// <param name="allocator">Allocator used for temporary tensor allocations during exponential and multiplication operations.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;
	};
}

#include "elementwiseGradFn.inl"