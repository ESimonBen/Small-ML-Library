/// convolution.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	/// <summary>
	/// Calculates the output size of a 1-D convolution for the given input size and convolution parameters.
	/// </summary>
	/// <param name="inputSize">Size of the input along the dimension being convolved (e.g., width or height).</param>
	/// <param name="kernelSize">Size (length) of the convolution kernel.</param>
	/// <param name="stride">Step size between successive kernel positions.</param>
	/// <param name="padding">Amount of padding added to each side of the input.</param>
	/// <param name="dilation">Dilation factor between kernel elements (1 means no dilation).</param>
	/// <returns>The computed output size as a size_t. Equivalent to (paddedInputSize - effectiveKernelSize) / stride + 1, where effectiveKernelSize = dilation * (kernelSize - 1) + 1 and paddedInputSize = inputSize + 2 * padding. Integer division (truncation) is used.</returns>
	size_t ComputeConvOutputSize(size_t inputSize, size_t kernelSize, size_t stride, size_t padding, size_t dilation);

	/// <summary>
	/// Performs a 1-dimensional convolution of an input tensor with a kernel tensor, optionally adding a bias. Validates tensor ranks, channel compatibility, allocator equality, and convolution parameters; may enable autograd for the result if any input requires gradients.
	/// </summary>
	/// <typeparam name="T">Numeric element type stored in the tensors (e.g., float, double).</typeparam>
	/// <param name="input">A 3-D tensor with shape {batchSize, inputChannels, inputLength}. Must use the same allocator as kernel (and bias if provided). Rank must be 3.</param>
	/// <param name="kernel">A 3-D tensor with shape {outputChannels, inputChannels, kernelLength}. Must use the same allocator as input. Rank must be 3 and inputChannels must match kernel's channel dimension.</param>
	/// <param name="bias">Optional pointer to a 1-D tensor of length outputChannels to add to each output channel. If non-null, must use the same allocator as input. May be nullptr to indicate no bias.</param>
	/// <param name="stride">Stride of the convolution (must be > 0).</param>
	/// <param name="padding">Number of zeros to pad on both sides of the input length dimension.</param>
	/// <param name="dilation">Dilation factor for the kernel (must be > 0).</param>
	/// <returns>A 3-D tensor with shape {batchSize, outputChannels, outputLength}, where outputLength = (inputLength + 2*padding - effectiveKernelLength) / stride + 1 and effectiveKernelLength = dilation * (kernelLength - 1) + 1. The returned tensor uses the input allocator. If input, kernel, or bias require gradients, the output will have gradient tracking enabled and a corresponding gradient function is attached.</returns>
	template <typename T>
	TensorCore::Tensor<T> Conv1D(const TensorCore::Tensor<T>& input, const TensorCore::Tensor<T>& kernel, const TensorCore::Tensor<T>* bias = nullptr,
								 size_t stride = 1, size_t padding = 0, size_t dilation = 1);

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
	/// <returns>A TensorCore::Tensor with shape [N, C_out, H_out, W_out], where H_out and W_out are computed from the input size, padding, effective kernel size (kernel and dilation), and stride. The output may be marked to require gradients and have a Conv2D gradient function attached if input, kernel, or bias require gradients. The function throws std::runtime_error for invalid ranks, allocator mismatches, channel mismatches, zero strides/dilations, zero kernel dimensions, or when the kernel is larger than the padded input.</returns>
	template <typename T>
	TensorCore::Tensor<T> Conv2D(const TensorCore::Tensor<T>& input, const TensorCore::Tensor<T>& kernel, const TensorCore::Tensor<T>* bias = nullptr,
								 size_t strideH = 1, size_t strideW = 1,
								 size_t paddingH = 0, size_t paddingW = 0,
								 size_t dilationH = 1, size_t dilationW = 1);

	/// <summary>
	/// Performs a 3D convolution of a 5-D input tensor with a 5-D kernel tensor and an optional bias, producing a 5-D output tensor. Validates shapes, allocators, and convolution parameters and attaches a gradient function if any input requires gradients.
	/// </summary>
	/// <typeparam name="T">Element type of the tensors (e.g., float, double).</typeparam>
	/// <param name="input">A 5-D tensor of type T with shape [batch, in_channels, depth, height, width]. Must use the same allocator as kernel (and bias if provided).</param>
	/// <param name="kernel">A 5-D tensor of type T with shape [out_channels, in_channels, kernelDepth, kernelHeight, kernelWidth]. in_channels must match input's channel dimension.</param>
	/// <param name="bias">Optional pointer to a 1-D tensor of type T of length out_channels. If provided, must use the same allocator as input and kernel. Can be nullptr to omit bias.</param>
	/// <param name="strideD">Stride along the depth dimension. Must be > 0.</param>
	/// <param name="strideH">Stride along the height dimension. Must be > 0.</param>
	/// <param name="strideW">Stride along the width dimension. Must be > 0.</param>
	/// <param name="paddingD">Padding applied to the depth dimension (non-negative).</param>
	/// <param name="paddingH">Padding applied to the height dimension (non-negative).</param>
	/// <param name="paddingW">Padding applied to the width dimension (non-negative).</param>
	/// <param name="dilationD">Dilation factor for the kernel along the depth dimension. Must be > 0.</param>
	/// <param name="dilationH">Dilation factor for the kernel along the height dimension. Must be > 0.</param>
	/// <param name="dilationW">Dilation factor for the kernel along the width dimension. Must be > 0.</param>
	/// <returns>A 5-D TensorCore::Tensor with shape [batch, out_channels, outputDepth, outputHeight, outputWidth], where each output spatial dimension is computed as ((inputDim + 2*padding - effectiveKernelDim) / stride) + 1. If any input (input, kernel, or bias) requires gradients, the returned tensor will be marked to require gradients and will have a Conv3D gradient function attached.</returns>
	template <typename T>
	TensorCore::Tensor<T> Conv3D(const TensorCore::Tensor<T>& input, const TensorCore::Tensor<T>& kernel, const TensorCore::Tensor<T>* bias = nullptr,
								 size_t strideD = 1, size_t strideH = 1, size_t strideW = 1,
								 size_t paddingD = 0, size_t paddingH = 0, size_t paddingW = 0,
								 size_t dilationD = 1, size_t dilationH = 1, size_t dilationW = 1);
}

#include "convolution.inl"