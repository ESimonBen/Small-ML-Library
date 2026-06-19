 /// reductionGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	/// <summary>
	/// A gradient function object that implements the backward pass for a sum operation. Inherits from GradFn<T> and performs gradient propagation/accumulation for tensors of type T.
	/// </summary>
	/// <typeparam name="T">The element type stored in tensors and used for gradient values (e.g., float, double, or other numeric types).</typeparam>
	template <typename T>
	class SumGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Initializes a SumGradFn<T> from an implementation pointer, forwarding the pointer to the GradFn<T> base and capturing the input shape.
		/// </summary>
		/// <typeparam name="T">The value type handled by the gradient function (e.g., the element type of tensors or gradients).</typeparam>
		/// <param name="a">A std::shared_ptr to a GradFn<T>::Impl that provides the underlying implementation. It is passed to the GradFn<T> base-class constructor and its shape (a->shape) is used to initialize inputShape.</param>
		SumGradFn(std::shared_ptr<typename GradFn<T>::Impl> a);

		/// <summary>
		/// Backward pass for a sum operation: propagates a scalar output gradient to the input by detaching and expanding it to the input shape, then calling the input's Backward.
		/// </summary>
		/// <typeparam name="T">Element type of the tensors (the numeric type of input and gradient values).</typeparam>
		/// <param name="gradOutput">Gradient tensor for the operation's output. Must be a scalar (NumElements() == 1). The function detaches this tensor and expands it to the original input shape before propagating.</param>
		/// <param name="allocator">Arena allocator used for temporary memory allocations during gradient construction/expansion (passed to any tensor allocation operations).</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator);

	private:
		Utils::Shape inputShape; /// Represents the input shape of the input tensor.
	};

	/// <summary>
	/// Gradient function object for the elementwise max operation. Stores context from the forward pass and computes gradients during the backward pass.
	/// </summary>
	/// <typeparam name="T">The element type of tensors (for example float or double).</typeparam>
	template <typename T>
	class MaxGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Initializes a MaxGradFn instance with the provided input gradient implementation and a maximum value used to clamp gradients.
		/// </summary>
		/// <typeparam name="T">The numeric type of the elements and gradients.</typeparam>
		/// <param name="a">Shared pointer to the input gradient function implementation (GradFn<T>::Impl). The constructor uses this to initialize the base GradFn and to obtain the input shape.</param>
		/// <param name="maxValue">Maximum value of type T used for clamping gradients.</param>
		MaxGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T maxValue);

		/// <summary>
		/// Performs the backward pass for a max operation: expects a scalar output gradient, splits that scalar gradient evenly among all input elements equal to the stored maximum, and propagates the resulting input gradient.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors (e.g., float, double, int).</typeparam>
		/// <param name="gradOutput">A tensor containing the gradient of the output. Must be a scalar (NumElements() == 1); otherwise the function throws std::runtime_error.</param>
		/// <param name="allocator">An ArenaAllocator used to allocate the temporary gradInput tensor.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		Utils::Shape inputShape; /// Represents the input shape of the input tensor.
		T maxValue; /// Holds the maximum value in the entire input tensor.
	};

	/// <summary>
	/// Gradient function object that performs the backward pass for an elementwise minimum operation against a fixed scalar and chains to a previous gradient implementation.
	/// </summary>
	/// <typeparam name="T">The element type of tensors (e.g., float, double, int).</typeparam>
	template <typename T>
	class MinGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs a MinGradFn<T> that applies a lower bound to gradient values.
		/// </summary>
		/// <typeparam name="T">The numeric type for gradients (e.g., float, double).</typeparam>
		/// <param name="a">Shared pointer to the input gradient function implementation (GradFn<T>::Impl). This is passed to the base GradFn and its shape is used to initialize inputShape.</param>
		/// <param name="minValue">The minimum value used to clamp gradients.</param>
		MinGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T minValue);

		/// <summary>
		/// Backpropagates a scalar gradient through a min reduction. The scalar gradient is distributed equally among all input elements equal to the minimum value, and then propagated to the input tensor.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors (e.g., float, double, int) used for values and gradients.</typeparam>
		/// <param name="gradOutput">A 1-element (scalar) gradient tensor for the result of the min operation. Must contain exactly one element; if not, the function throws std::runtime_error. The single scalar value is read and divided among the input elements equal to the minimum.</param>
		/// <param name="allocator">Allocator used to allocate the gradient tensor for the input (used when constructing gradInput).</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		Utils::Shape inputShape; /// Represents the input shape of the input tensor.
		T minValue; /// Holds the minimum value in the entire input tensor.
	};

	/// <summary>
	/// Gradient function object that computes and propagates gradients for a sum reduction performed along a specific axis.
	/// </summary>
	/// <typeparam name="T">The element type of the tensors (for example float, double, or an integral type).</typeparam>
	template <typename T>
	class AxisSumGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs an AxisSumGradFn that computes the gradient for a sum reduction along a specified axis.
		/// </summary>
		/// <typeparam name="T">The element type used by the gradient function and associated tensors.</typeparam>
		/// <param name="a">A shared pointer to the upstream GradFn<T>::Impl. Used to initialize the base GradFn and to obtain the input shape (a->shape).</param>
		/// <param name="axis">The axis index along which the sum reduction was performed. Must be less than the input shape rank.</param>
		/// <param name="keepDims">If true, reduced dimensions are kept with size 1; if false, reduced dimensions are removed.</param>
		AxisSumGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis, bool keepDims);

		/// <summary>
		/// Performs the backward pass for an axis-wise sum operation, reconstructing and propagating the gradient to the input tensor.
		/// </summary>
		/// <typeparam name="T">Element type of the tensors (e.g., float, double) handled by the function.</typeparam>
		/// <param name="gradOutput">Gradient tensor with respect to the operation's output. It is detached and then reshaped or expanded as needed to match the input shape before propagation.</param>
		/// <param name="allocator">Allocator used for temporary memory operations (for example, when unsqueezing or otherwise reshaping intermediate tensors).</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		Utils::Shape inputShape; /// Represents the input shape of the input tensor.
		size_t m_Axis; /// Represents the axis that was summed.
		bool m_KeepDims; /// Boolean flag that indicates whether to preserve dimensions.
	};

	/// <summary>
	/// Gradient function object that computes and propagates gradients for a max reduction performed along a specific axis.
	/// </summary>
	/// <typeparam name="T">Element type of the input and output tensors (e.g., float, double, int).</typeparam>
	template <typename T>
	class AxisMaxGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs an AxisMaxGradFn<T> that computes the gradient for an axis-wise maximum operation.
		/// </summary>
		/// <typeparam name="T">The element type used by the gradient function (e.g., float, double).</typeparam>
		/// <param name="a">Shared pointer to the implementation of the preceding gradient function (std::shared_ptr<GradFn<T>::Impl>).</param>
		/// <param name="axis">Index of the axis along which the max operation was performed.</param>
		/// <param name="keepDims">If true, reduced dimensions were kept with size 1; if false, they were removed.</param>
		AxisMaxGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis, bool keepDims);

		/// <summary>
		/// Performs the backward pass for an AxisMax operation: computes the gradient with respect to the input by distributing the output gradient to positions equal to the axis-wise maximum and then invokes the input's Backward to propagate that gradient.
		/// </summary>
		/// <typeparam name="T">Element type of the tensors (for example float or double) handled by this specialization.</typeparam>
		/// <param name="gradOutput">Gradient tensor with respect to the output of the AxisMax operation. If the output was reduced (m_KeepDims == false), this function will unsqueeze/expand gradOutput as needed before distributing it back to the input shape.</param>
		/// <param name="allocator">Arena allocator used to allocate temporary tensors and buffers during the backward computation.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		size_t m_Axis; /// Represents the axis that was checked for a maximum
		bool m_KeepDims; /// Boolean flag that indicates whether to preserve dimensions.
	};

	/// <summary>
	/// Gradient function object that computes and applies the backward pass for a reduction that takes the minimum along a specified axis.
	/// </summary>
	/// <typeparam name="T">The scalar type of tensor elements (for example float, double, or an integral type) used by the tensors and gradients.</typeparam>
	template <typename T>
	class AxisMinGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Initializes an AxisMinGradFn<T> that computes gradients for a minimum reduction along a specified axis.
		/// </summary>
		/// <typeparam name="T">The element/data type used by the gradient function and associated tensors.</typeparam>
		/// <param name="a">A shared pointer to the underlying GradFn<T>::Impl that this gradient function wraps or uses as its input.</param>
		/// <param name="axis">The index of the axis along which the minimum reduction was performed.</param>
		/// <param name="keepDims">If true, keep reduced dimensions with size 1; if false, remove reduced dimensions.</param>
		AxisMinGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis, bool keepDims);

		/// <summary>
		/// Performs the backward pass for an axis-wise minimum operation: constructs a mask of elements equal to the axis minimum, expands and distributes the incoming gradient among tied minima, and propagates the resulting gradient to the input tensor. If the input does not require gradients, no action is taken.
		/// </summary>
		/// <typeparam name="T">The scalar element type of the tensors (for example, float or double).</typeparam>
		/// <param name="gradOutput">Gradient tensor from the next layer with respect to the output of the axis-min operation. May be unsqueezed if the original reduction removed dimensions.</param>
		/// <param name="allocator">Arena allocator used for temporary tensor allocations during the gradient computation.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		size_t m_Axis; /// Represents the axis that was checked for a minimum
		bool m_KeepDims; /// Boolean flag that indicates whether to preserve dimensions.
	};
}

#include "reductionGradFn.inl"