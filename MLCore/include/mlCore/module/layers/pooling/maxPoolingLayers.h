/// maxPoolingLayers.h
#pragma once
#include <mlCore/module/module.h>
#include <mlCore/parameters/initialization.h>

namespace MLCore::NN {
	/// <summary>
	/// A 1D max-pooling layer module that applies a sliding-window maximum over an input tensor.
	/// </summary>
	/// <typeparam name="T">The numeric data type used by the layer's tensors (for example float or double).</typeparam>
	template <typename T>
	class MaxPool1DLayer : public Module<T> {
	public:
		/// <summary>
		/// Constructs a MaxPool1DLayer with the specified filter length, stride, padding, dilation, and ceil mode.
		/// </summary>
		/// <typeparam name="T">Data type of the layer's elements (for example float or double).</typeparam>
		/// <param name="filterLength">Length of the pooling filter (kernel size).</param>
		/// <param name="stride">Step size between successive pooling windows.</param>
		/// <param name="padding">Number of padding elements added to each side of the input.</param>
		/// <param name="dilation">Spacing between elements within the filter (dilation factor).</param>
		/// <param name="ceilMode">If true, use ceil when computing the output size; otherwise use floor.</param>
		MaxPool1DLayer(size_t filterLength, size_t stride = 0, size_t padding = 0, size_t dilation = 1, bool ceilMode = false);

		/// <summary>
		/// Performs the forward pass of a 1D max-pooling layer on the provided input tensor.
		/// </summary>
		/// <typeparam name="T">The element type of the tensor (e.g., float, double).</typeparam>
		/// <param name="input">The input tensor to be pooled (passed by const reference).</param>
		/// <returns>A tensor containing the result of applying 1D max pooling to the input using the layer's configured filter length, stride, padding, dilation, and ceil-mode settings.</returns>
		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) override;

	private:
		size_t m_FilterLength; /// Stores the length of the filter.
		size_t m_Stride; /// Stride along the length for the pooling operation.
		size_t m_Padding; /// Padding along the length for the pooling operation.
		size_t m_Dilation; /// Dilation along the length for the pooling operation.
		bool m_CeilMode; /// Flag indicating whether ceiling mode is enabled.
	};

	/// <summary>
	/// A 2D max-pooling layer that performs spatial downsampling by taking the maximum value over sliding windows.
	/// </summary>
	/// <typeparam name="T">Element type of the input and output tensors (for example float or double).</typeparam>
	template <typename T>
	class MaxPool2DLayer : public Module<T> {
	public:
		/// <summary>
		/// Constructs a 2D max-pooling layer configured with the specified kernel size, strides, padding, dilation, and ceil mode.
		/// </summary>
		/// <param name="filterHeight">The height of the pooling filter (kernel).</param>
		/// <param name="filterWidth">The width of the pooling filter (kernel).</param>
		/// <param name="strideH">Vertical stride (step) for the pooling operation. Default is 0.</param>
		/// <param name="strideW">Horizontal stride (step) for the pooling operation. Default is 0.</param>
		/// <param name="paddingH">Vertical padding applied to the input. Default is 0.</param>
		/// <param name="paddingW">Horizontal padding applied to the input. Default is 0.</param>
		/// <param name="dilationH">Vertical dilation (spacing) between elements in the filter. Default is 1.</param>
		/// <param name="dilationW">Horizontal dilation (spacing) between elements in the filter. Default is 1.</param>
		/// <param name="ceilMode">If true, use ceil to compute the output shape; otherwise use floor. Default is false.</param>
		MaxPool2DLayer(size_t filterHeight, size_t filterWidth, size_t strideH = 0, size_t strideW = 0, 
					   size_t paddingH = 0, size_t paddingW = 0, size_t dilationH = 1, size_t dilationW = 1, bool ceilMode = false);

		/// <summary>
		/// Performs the forward pass of a 2D max-pooling layer, applying max pooling to the input tensor using the layer's configured filter size, strides, padding, dilation, and ceil mode.
		/// </summary>
		/// <typeparam name="T">The element type of the tensor (e.g., float, double, int).</typeparam>
		/// <param name="input">The input tensor to be pooled (const reference).</param>
		/// <returns>A TensorCore::Tensor containing the result of the 2D max-pooling operation.</returns>
		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) override;

	private:
		size_t m_FilterHeight, m_FilterWidth; /// Stores the height and width of the filter.
		size_t m_StrideH, m_StrideW; /// Stride along the height and width for the pooling operation.
		size_t m_PaddingH, m_PaddingW; /// Padding along the height and width for the pooling operation.
		size_t m_DilationH, m_DilationW; /// Dilation along the height and width for the pooling operation.
		bool m_CeilMode; /// Flag indicating whether ceiling mode is enabled.
	};

	/// <summary>
	/// Layer that performs 3D max pooling over an input tensor.
	/// </summary>
	/// <typeparam name="T">The numeric data type of the elements stored in the input and output tensors (e.g., float, double).</typeparam>
	template <typename T>
	class MaxPool3DLayer : public Module<T> {
	public:
		/// <summary>
		/// Constructs a MaxPool3DLayer initialized with the given 3D pooling kernel sizes, strides, paddings, dilations, and ceil mode.
		/// </summary>
		/// <typeparam name="T">Numeric type used for the layer's computations and stored values (e.g., float, double).</typeparam>
		/// <param name="filterDepth">Depth of the pooling filter (kernel) in elements.</param>
		/// <param name="filterHeight">Height of the pooling filter (kernel) in elements.</param>
		/// <param name="filterWidth">Width of the pooling filter (kernel) in elements.</param>
		/// <param name="strideD">Stride (step) along the depth dimension.</param>
		/// <param name="strideH">Stride (step) along the height dimension.</param>
		/// <param name="strideW">Stride (step) along the width dimension.</param>
		/// <param name="paddingD">Padding applied to the depth dimension (number of elements added on each side).</param>
		/// <param name="paddingH">Padding applied to the height dimension (number of elements added on each side).</param>
		/// <param name="paddingW">Padding applied to the width dimension (number of elements added on each side).</param>
		/// <param name="dilationD">Dilation (spacing) between elements in the filter along the depth dimension.</param>
		/// <param name="dilationH">Dilation (spacing) between elements in the filter along the height dimension.</param>
		/// <param name="dilationW">Dilation (spacing) between elements in the filter along the width dimension.</param>
		/// <param name="ceilMode">If true, output spatial dimensions are computed using ceil when dividing; otherwise floor is used.</param>
		MaxPool3DLayer(size_t filterDepth, size_t filterHeight, size_t filterWidth,
					   size_t strideD = 0, size_t strideH = 0, size_t strideW = 0,
					   size_t paddingD = 0, size_t paddingH = 0, size_t paddingW = 0, 
					   size_t dilationD = 1, size_t dilationH = 1, size_t dilationW = 1, bool ceilMode = false);

		/// <summary>
		/// Performs the forward pass of a 3D max-pooling layer on the given input tensor using the layer's configured filter size, strides, padding, dilation, and ceil-mode.
		/// </summary>
		/// <typeparam name="T">Element type of the tensor (for example float, double, or int).</typeparam>
		/// <param name="input">Input tensor to be pooled (e.g., layout: batch, channels, depth, height, width).</param>
		/// <returns>A tensor containing the result of applying 3D max pooling to the input. The output shape is determined by the layer's filter dimensions, strides, paddings, dilations, and ceil-mode.</returns>
		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) override;

	private:
		size_t m_FilterDepth, m_FilterHeight, m_FilterWidth; /// Stores the depth, height and width of the filter.
		size_t m_StrideD, m_StrideH, m_StrideW; /// Stride along the depth, height and width for the pooling operation.
		size_t m_PaddingD, m_PaddingH, m_PaddingW; /// Padding along the depth, height and width for the pooling operation.
		size_t m_DilationD, m_DilationH, m_DilationW; /// Dilation along the depth, height and width for the pooling operation.
		bool m_CeilMode; /// Flag indicating whether ceiling mode is enabled.
	};
}

#include "maxPoolingLayers.inl"