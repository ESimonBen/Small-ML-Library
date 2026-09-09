/// pooling.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	/// <summary>
	/// Calculates the output size for a pooling operation along a single dimension.
	/// </summary>
	/// <param name="inputSize">Size of the input dimension (e.g., width or height).</param>
	/// <param name="filterSize">Size of the pooling filter (kernel).</param>
	/// <param name="stride">Stride (step) of the pooling operation.</param>
	/// <param name="padding">Amount of padding added to both sides of the input.</param>
	/// <param name="dilation">Dilation factor for the filter; effective kernel size is computed as dilation * (filterSize - 1) + 1.</param>
	/// <param name="ceilMode">If true, uses ceiling when dividing to compute the output size; otherwise uses floor (standard integer division).</param>
	/// <returns>The computed output size (number of positions) for the pooled dimension.</returns>
	size_t ComputePoolOutputSize(size_t inputSize, size_t filterSize, size_t stride, size_t padding, size_t dilation, bool ceilMode);

	/// <summary>
	/// Performs 1D max pooling over the last dimension of a 3D tensor (batch, channels, length). For each sliding window of length filterLength (with given stride, padding and dilation) it selects the maximum value. The output length is computed with ComputePoolOutputSize. If the input requires gradients, the operator records the indices of the maxima and configures the returned tensor to require gradients.
	/// </summary>
	/// <typeparam name="T">Element type of the input and output tensors (numeric type stored by TensorCore::Tensor).</typeparam>
	/// <param name="input">A 3D input tensor with shape {batchSize, channels, inputLength}. The function throws std::runtime_error if input.Rank() != 3.</param>
	/// <param name="filterLength">The size of the pooling window (must be > 0).</param>
	/// <param name="stride">The step between successive pooling windows (must be > 0).</param>
	/// <param name="padding">Amount of implicit padding applied to both sides of the input along the length dimension.</param>
	/// <param name="dilation">Spacing between elements within the pooling window (must be > 0).</param>
	/// <param name="ceilMode">If true, use ceil when computing output length; otherwise use floor (affects ComputePoolOutputSize).</param>
	/// <returns>A TensorCore::Tensor of shape {batchSize, channels, outputLength} containing the maximum values from each pooling window along the last dimension. If input.RequiresGrad() was true, the returned tensor is marked to require gradients and a backward function is attached; in that case the implementation also records the flattened input indices of the maxima for use in the gradient computation.</returns>
	template <typename T>
	TensorCore::Tensor<T> MaxPool1D(const TensorCore::Tensor<T>& input, size_t filterLength, size_t stride = 0, size_t padding = 0, size_t dilation = 1, bool ceilMode = false);

	/// <summary>
	/// Performs 1D minimum pooling on the input tensor by applying MaxPool1D to the negated input.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor values.</typeparam>
	/// <param name="input">Constant reference to the input tensor to be pooled.</param>
	/// <param name="filterLength">Length of the pooling window (number of elements in the filter).</param>
	/// <param name="stride">Stride (step) between consecutive pooling windows.</param>
	/// <param name="padding">Amount of padding applied to both sides of the input (in elements).</param>
	/// <param name="dilation">Dilation factor for the pooling window (spacing between elements in the filter).</param>
	/// <param name="ceilMode">If true, use ceiling when computing output dimensions; if false, use floor.</param>
	/// <returns>A tensor of type TensorCore::Tensor containing the min-pooled result. The output shape is determined by filterLength, stride, padding, dilation, and ceilMode.</returns>
	template <typename T>
	TensorCore::Tensor<T> MinPool1D(const TensorCore::Tensor<T>& input, size_t filterLength, size_t stride = 0, size_t padding = 0, size_t dilation = 1, bool ceilMode = false);

	/// <summary>
	/// Performs 2D max pooling on a 4-D tensor (batch, channels, height, width). Computes the maximum value in each pooling window and returns a tensor of pooled values. If the input requires gradients, argmax indices are recorded and the returned tensor is configured for backpropagation.
	/// </summary>
	/// <typeparam name="T">Element type of the input and output tensors (numeric type).</typeparam>
	/// <param name="input">Input tensor of rank 4 with shape [batch, channels, height, width]. May require gradients; if so, argmax indices are stored for use in the backward pass.</param>
	/// <param name="filterHeight">Height of the pooling filter (must be > 0).</param>
	/// <param name="filterWidth">Width of the pooling filter (must be > 0).</param>
	/// <param name="strideH">Vertical stride (must be > 0).</param>
	/// <param name="strideW">Horizontal stride (must be > 0).</param>
	/// <param name="paddingH">Vertical padding applied to the input (can be 0).</param>
	/// <param name="paddingW">Horizontal padding applied to the input (can be 0).</param>
	/// <param name="dilationH">Vertical dilation factor for the filter (must be > 0).</param>
	/// <param name="dilationW">Horizontal dilation factor for the filter (must be > 0).</param>
	/// <param name="ceilMode">If true, use ceil when computing the output spatial size; if false, use floor.</param>
	/// <returns>A TensorCore::Tensor containing the pooled output with shape [batch, channels, outputHeight, outputWidth], where outputHeight and outputWidth are computed by ComputePoolOutputSize. If the input required gradients, the returned tensor will require gradients and carry a gradient function that uses stored argmax indices.</returns>
	template <typename T>
	TensorCore::Tensor<T> MaxPool2D(const TensorCore::Tensor<T>& input, size_t filterHeight, size_t filterWidth,
									size_t strideH = 0, size_t strideW = 0, size_t paddingH = 0, size_t paddingW = 0, size_t dilationH = 1, size_t dilationW = 1, bool ceilMode = false);

	/// <summary>
	/// Computes 2D minimum pooling on the input tensor using the specified filter size, strides, padding, dilation, and ceil mode. This implementation obtains minima by negating the input, applying max pooling, and negating that result.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor (e.g., float, double, int).</typeparam>
	/// <param name="input">Input tensor to apply min pooling to.</param>
	/// <param name="filterHeight">Height of the pooling filter (kernel).</param>
	/// <param name="filterWidth">Width of the pooling filter (kernel).</param>
	/// <param name="strideH">Vertical stride (step) between pooling windows.</param>
	/// <param name="strideW">Horizontal stride (step) between pooling windows.</param>
	/// <param name="paddingH">Vertical padding applied to the input.</param>
	/// <param name="paddingW">Horizontal padding applied to the input.</param>
	/// <param name="dilationH">Vertical dilation (spacing) between kernel elements.</param>
	/// <param name="dilationW">Horizontal dilation (spacing) between kernel elements.</param>
	/// <param name="ceilMode">If true, use ceiling when computing output dimensions for uneven divisions due to padding; if false, use floor.</param>
	/// <returns>A tensor of type Tensor containing the minimum values computed over each pooling window.</returns>
	template <typename T>
	TensorCore::Tensor<T> MinPool2D(const TensorCore::Tensor<T>& input, size_t filterHeight, size_t filterWidth,
									size_t strideH = 0, size_t strideW = 0, size_t paddingH = 0, size_t paddingW = 0, size_t dilationH = 1, size_t dilationW = 1, bool ceilMode = false);

	/// <summary>
	/// Performs 3D max pooling on a 5-D tensor (batch, channels, depth, height, width). Computes pooled output dimensions using the provided filter size, strides, paddings, dilations and ceilMode. If the input requires gradients, records argmax indices and wires a gradient function for backward propagation. Throws std::runtime_error on invalid input shape or zero-valued stride/dilation/filter dimensions.
	/// </summary>
	/// <typeparam name="T">Element type stored in the input and output tensors (e.g., float, double). Also used as the element type for stored argmax indices when gradients are required.</typeparam>
	/// <param name="input">Input tensor of rank 5 with shape {batch, channels, depth, height, width}. Must be 5-D or a runtime_error is thrown. If input.RequiresGrad() is true, argmax indices are recorded to support the backward pass.</param>
	/// <param name="filterDepth">Depth of the pooling kernel (must be > 0).</param>
	/// <param name="filterHeight">Height of the pooling kernel (must be > 0).</param>
	/// <param name="filterWidth">Width of the pooling kernel (must be > 0).</param>
	/// <param name="strideD">Stride along the depth dimension (must be > 0).</param>
	/// <param name="strideH">Stride along the height dimension (must be > 0).</param>
	/// <param name="strideW">Stride along the width dimension (must be > 0).</param>
	/// <param name="paddingD">Zero-padding added to both sides of the depth dimension.</param>
	/// <param name="paddingH">Zero-padding added to both sides of the height dimension.</param>
	/// <param name="paddingW">Zero-padding added to both sides of the width dimension.</param>
	/// <param name="dilationD">Dilation factor along the depth dimension (must be > 0).</param>
	/// <param name="dilationH">Dilation factor along the height dimension (must be > 0).</param>
	/// <param name="dilationW">Dilation factor along the width dimension (must be > 0).</param>
	/// <param name="ceilMode">If true, use ceil when computing output spatial dimensions; otherwise use floor. Affects how output size is computed by ComputePoolOutputSize().</param>
	/// <returns>A TensorCore::Tensor containing the pooled output with shape {batch, channels, outputDepth, outputHeight, outputWidth}. Each element is the maximum value over the corresponding receptive field. If the input required gradients, the returned tensor will have requires_grad set and a backward function attached.</returns>
	template <typename T>
	TensorCore::Tensor<T> MaxPool3D(const TensorCore::Tensor<T>& input, size_t filterDepth, size_t filterHeight, size_t filterWidth,
									size_t strideD = 0, size_t strideH = 0, size_t strideW = 0,
									size_t paddingD = 0, size_t paddingH = 0, size_t paddingW = 0,
									size_t dilationD = 1, size_t dilationH = 1, size_t dilationW = 1, bool ceilMode = false);

	/// <summary>
	/// Performs 3D minimum pooling on the input tensor. Implemented by negating the input, applying MaxPool3D, then negating the result.
	/// </summary>
	/// <typeparam name="T">Element type stored in the input and output tensors.</typeparam>
	/// <param name="input">The input tensor to be pooled (const reference).</param>
	/// <param name="filterDepth">Depth of the pooling window.</param>
	/// <param name="filterHeight">Height of the pooling window.</param>
	/// <param name="filterWidth">Width of the pooling window.</param>
	/// <param name="strideD">Stride (step) along the depth dimension.</param>
	/// <param name="strideH">Stride (step) along the height dimension.</param>
	/// <param name="strideW">Stride (step) along the width dimension.</param>
	/// <param name="paddingD">Padding applied to the depth dimension.</param>
	/// <param name="paddingH">Padding applied to the height dimension.</param>
	/// <param name="paddingW">Padding applied to the width dimension.</param>
	/// <param name="dilationD">Dilation factor for the pooling window along the depth dimension.</param>
	/// <param name="dilationH">Dilation factor for the pooling window along the height dimension.</param>
	/// <param name="dilationW">Dilation factor for the pooling window along the width dimension.</param>
	/// <param name="ceilMode">If true, use ceiling when computing output spatial dimensions; otherwise use floor.</param>
	/// <returns>A TensorCore::Tensor containing the result of 3D min pooling using the specified filter size, strides, paddings, dilations, and ceilMode.</returns>
	template <typename T>
	TensorCore::Tensor<T> MinPool3D(const TensorCore::Tensor<T>& input, size_t filterDepth, size_t filterHeight, size_t filterWidth,
									size_t strideD = 0, size_t strideH = 0, size_t strideW = 0,
									size_t paddingD = 0, size_t paddingH = 0, size_t paddingW = 0,
									size_t dilationD = 1, size_t dilationH = 1, size_t dilationW = 1, bool ceilMode = false);
}

#include "pooling.inl"