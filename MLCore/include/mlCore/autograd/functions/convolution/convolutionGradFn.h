/// convolutionGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
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
					 size_t strideH, size_t strideW, size_t paddingH, size_t paddingW, size_t dilationH, size_t dilationW);

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
}

#include "convolutionGradFn.inl"