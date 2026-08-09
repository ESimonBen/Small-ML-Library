/// convolution.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	/// <summary>
	/// Performs a 2D convolution of input with kernel and optional bias and returns the resulting output tensor. Validates ranks, channel compatibility, allocator compatibility, strides, dilations, and kernel/padding sizes. If any operand requires gradients, the output will have a gradient function attached.
	/// </summary>
	/// <typeparam name="T">Element type of the tensors (e.g., float, double).</typeparam>
	/// <param name="input">Input tensor with shape [N, C_in, H, W]. Must have rank 4 and use the same allocator as kernel (and bias if provided).</param>
	/// <param name="kernel">Kernel tensor with shape [C_out, C_in, K_h, K_w]. Must have rank 4 and input channel dimension must match kernel's C_in.</param>
	/// <param name="bias">Optional pointer to bias tensor with shape [C_out]. If non-null, must share the same allocator and is added per output channel.</param>
	/// <param name="strideH">Vertical stride (must be > 0).</param>
	/// <param name="strideW">Horizontal stride (must be > 0).</param>
	/// <param name="paddingH">Vertical padding applied to the input (number of rows padded on each side).</param>
	/// <param name="paddingW">Horizontal padding applied to the input (number of columns padded on each side).</param>
	/// <param name="dilationH">Vertical dilation factor for the kernel (must be > 0).</param>
	/// <param name="dilationW">Horizontal dilation factor for the kernel (must be > 0).</param>
	/// <returns>A TensorCore::Tensor<T> with shape [N, C_out, H_out, W_out], where H_out and W_out are computed from the input size, padding, effective kernel size (kernel and dilation), and stride. The output may be marked to require gradients and have a Conv2D gradient function attached if input, kernel, or bias require gradients. The function throws std::runtime_error for invalid ranks, allocator mismatches, channel mismatches, zero strides/dilations, zero kernel dimensions, or when the kernel is larger than the padded input.</returns>
	template <typename T>
	TensorCore::Tensor<T> Conv2D(const TensorCore::Tensor<T>& input, const TensorCore::Tensor<T>& kernel, const TensorCore::Tensor<T>* bias = nullptr,
								 size_t strideH = 1, size_t strideW = 1, size_t paddingH = 0, size_t paddingW = 0, size_t dilationH = 1, size_t dilationW = 1);
}

#include "convolution.inl"