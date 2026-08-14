/// convolutionGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	/// <summary>
	/// Gradient function object for 1D convolution operations. Holds configuration for stride, padding, and dilation and performs backward propagation to compute and propagate gradients for input, kernel, and bias.
	/// </summary>
	/// <typeparam name="T">Element type of the tensors and gradients (e.g., float, double) used by this gradient function.</typeparam>
	template <typename T>
	class Conv1DGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs a Conv1DGradFn instance and initializes convolution gradient parameters.
		/// </summary>
		/// <typeparam name="T">The numeric type of the tensor elements (e.g., float, double) used by the gradient functions.</typeparam>
		/// <param name="input">Shared pointer to the GradFn<T>::Impl for the input tensor (gradient source for input).</param>
		/// <param name="kernel">Shared pointer to the GradFn<T>::Impl for the convolution kernel (gradient source for kernel).</param>
		/// <param name="bias">Shared pointer to the GradFn<T>::Impl for the bias term (gradient source for bias).</param>
		/// <param name="stride">Stride size for the 1D convolution.</param>
		/// <param name="padding">Padding size applied to the input.</param>
		/// <param name="dilation">Dilation factor for the convolution kernel.</param>
		Conv1DGradFn(std::shared_ptr<typename GradFn<T>::Impl> input, std::shared_ptr<typename GradFn<T>::Impl> kernel, std::shared_ptr<typename GradFn<T>::Impl> bias,
					 size_t stride, size_t padding, size_t dilation);

		/// <summary>
		/// Compute and apply gradients for a 1D convolution layer given the output gradient. Accumulates gradients for input, kernel, and optional bias and invokes Backward on each required input. Validates inputs, tensor ranks, and dimension consistency and throws on errors.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors (for example, float or double).</typeparam>
		/// <param name="gradOutput">The gradient tensor with respect to the layer output. Must be a rank-3 tensor with dimensions [batchSize, outputChannels, outputLength]. Used to compute and accumulate gradients for input, kernel, and optional bias. The function validates batch size and output channel count against the stored input and kernel tensors and will throw std::runtime_error on null inputs, rank mismatches, or dimension mismatches.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		size_t m_Stride; /// Stride along the length for the convolution operation.
		size_t m_Padding; /// Padding along the length for the convolution operation.
		size_t m_Dilation; /// Dilation along the length for the convolution operation.
	};

	/// <summary>
	/// Gradient function for a 2D convolution operation. Stores convolution hyperparameters (stride, padding, dilation) and implements Backward to compute and propagate gradients to the input, kernels, and bias.
	/// </summary>
	/// <typeparam name="T">Element type of the tensors (e.g., float, double) used by the GradFn and Tensor implementations.</typeparam>
	template <typename T>
	class Conv2DGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Initializes a Conv2DGradFn instance with the given gradient-function implementations and convolution parameters (stride, padding, dilation).
		/// </summary>
		/// <typeparam name="T">The element type used by the gradient functions (e.g., float, double) for the Conv2DGradFn template.</typeparam>
		/// <param name="input">Shared pointer to the GradFn<T>::Impl that computes gradients w.r.t. the input tensor.</param>
		/// <param name="kernel">Shared pointer to the GradFn<T>::Impl that computes gradients w.r.t. the convolution kernels.</param>
		/// <param name="bias">Shared pointer to the GradFn<T>::Impl that computes gradients w.r.t. the bias (may be null if no bias).</param>
		/// <param name="strideH">Vertical stride (number of rows to step between kernel applications).</param>
		/// <param name="strideW">Horizontal stride (number of columns to step between kernel applications).</param>
		/// <param name="paddingH">Vertical padding applied to the input (in rows).</param>
		/// <param name="paddingW">Horizontal padding applied to the input (in columns).</param>
		/// <param name="dilationH">Vertical dilation (spacing between kernel elements along rows).</param>
		/// <param name="dilationW">Horizontal dilation (spacing between kernel elements along columns).</param>
		Conv2DGradFn(std::shared_ptr<typename GradFn<T>::Impl> input, std::shared_ptr<typename GradFn<T>::Impl> kernel, std::shared_ptr<typename GradFn<T>::Impl> bias,
					 size_t strideH, size_t strideW,
					 size_t paddingH, size_t paddingW,
					 size_t dilationH, size_t dilationW);

		/// <summary>
		/// Computes and propagates gradients for a 2D convolution. Given the gradient of the convolution output, this method computes gradients with respect to the input, kernel, and optional bias (only if those tensors require gradients), allocates and accumulates the corresponding gradient tensors, and calls Backward on the original inputs to propagate gradients. It validates tensor dimensionality, channel counts, and batch size, and returns early if no gradients are required.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors (e.g., float).</typeparam>
		/// <param name="gradOutput">A 4-D tensor [N, C_out, H_out, W_out] containing the gradient of the convolution output. Must have 4 dimensions, matching batch size and output channel count expected by the stored input and kernel tensors. Elements are of type T.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		size_t m_StrideH, m_StrideW; /// Stride along height and width dimensions for the convolution operation.

		size_t m_PaddingH, m_PaddingW; /// Padding along height and width dimensions for the convolution operation.

		size_t m_DilationH, m_DilationW; /// Dilation along height and width dimensions for the convolution operation.
	};

	/// <summary>
	/// Gradient function for 3D convolution operations. Responsible for computing and propagating gradients for the input, kernel, and bias during backpropagation.
	/// </summary>
	/// <typeparam name="T">The numeric type of tensor elements (e.g., float, double) used by the convolution and its gradients.</typeparam>
	template <typename T>
	class Conv3DGradFn : public GradFn<T> {
	public:
		/// <summary>
		/// Constructs a Conv3DGradFn<T> that represents the gradient computation for a 3D convolution, initializing the associated input, kernel, and bias gradient function implementations and storing per-dimension stride, padding, and dilation parameters.
		/// </summary>
		/// <typeparam name="T">The numeric data type used by the gradient functions (for example, float or double).</typeparam>
		/// <param name="input">std::shared_ptr<GradFn<T>::Impl> pointing to the gradient-function implementation for the input tensor.</param>
		/// <param name="kernel">std::shared_ptr<GradFn<T>::Impl> pointing to the gradient-function implementation for the convolution kernel (weights).</param>
		/// <param name="bias">std::shared_ptr<GradFn<T>::Impl> pointing to the gradient-function implementation for the bias term.</param>
		/// <param name="strideD">Stride size along the depth dimension (size_t).</param>
		/// <param name="strideH">Stride size along the height dimension (size_t).</param>
		/// <param name="strideW">Stride size along the width dimension (size_t).</param>
		/// <param name="paddingD">Padding applied on the depth dimension (size_t).</param>
		/// <param name="paddingH">Padding applied on the height dimension (size_t).</param>
		/// <param name="paddingW">Padding applied on the width dimension (size_t).</param>
		/// <param name="dilationD">Dilation (spacing) of kernel elements along the depth dimension (size_t).</param>
		/// <param name="dilationH">Dilation (spacing) of kernel elements along the height dimension (size_t).</param>
		/// <param name="dilationW">Dilation (spacing) of kernel elements along the width dimension (size_t).</param>
		Conv3DGradFn(std::shared_ptr<typename GradFn<T>::Impl> input, std::shared_ptr<typename GradFn<T>::Impl> kernel, std::shared_ptr<typename GradFn<T>::Impl> bias,
					 size_t strideD, size_t strideH, size_t strideW,
					 size_t paddingD, size_t paddingH, size_t paddingW,
					 size_t dilationD, size_t dilationH, size_t dilationW);

		/// <summary>
		/// Computes and accumulates gradients for a 3D convolution and propagates them to the input, kernel, and optional bias tensors. Validates inputs and shapes before computation and may throw runtime_error on null inputs or dimension mismatches.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors (for example float or double).</typeparam>
		/// <param name="gradOutput">The output gradient tensor. Must be a 5-D tensor with shape [batchSize, outputChannels, outputDepth, outputHeight, outputWidth]. Used to compute gradients for the input, kernel, and optional bias; its batch size and output channel count are validated against the corresponding input and kernel dimensions.</param>
		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		size_t m_StrideD, m_StrideH, m_StrideW; /// Stride along depth, height and width dimensions for the convolution operation.

		size_t m_PaddingD, m_PaddingH, m_PaddingW; /// Padding along depth, height and width dimensions for the convolution operation.

		size_t m_DilationD, m_DilationH, m_DilationW; /// Dilation along depth, height and width dimensions for the convolution operation.
	};
}

#include "convolutionGradFn.inl"