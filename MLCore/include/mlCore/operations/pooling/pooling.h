/// pooling.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	size_t ComputePoolOutputSize(size_t inputSize, size_t filterSize, size_t stride, size_t padding, size_t dilation, bool ceilMode);

	template <typename T>
	TensorCore::Tensor<T> MaxPool1D(const TensorCore::Tensor<T>& input, size_t filterLength, size_t stride, size_t padding = 0, size_t dilation = 1, bool ceilMode = false);

	template <typename T>
	TensorCore::Tensor<T> MaxPool2D(const TensorCore::Tensor<T>& input, size_t filterHeight, size_t filterWidth,
									size_t strideH, size_t strideW, size_t paddingH = 0, size_t paddingW = 0, size_t dilationH = 1, size_t dilationW = 1, bool ceilMode = false);

	template <typename T>
	TensorCore::Tensor<T> MaxPool3D(const TensorCore::Tensor<T>& input, size_t filterDepth, size_t filterHeight, size_t filterWidth,
									size_t strideD, size_t strideH, size_t strideW,
									size_t paddingD = 0, size_t paddingH = 0, size_t paddingW = 0,
									size_t dilationD = 1, size_t dilationH = 1, size_t dilationW = 1, bool ceilMode = false);

	/// Overloaded versions with default strides (uses filter size as default stride)

	template <typename T>
	TensorCore::Tensor<T> MaxPool1D(const TensorCore::Tensor<T>& input, size_t filterLength, size_t padding = 0, size_t dilation = 1, bool ceilMode = false);

	template <typename T>
	TensorCore::Tensor<T> MaxPool2D(const TensorCore::Tensor<T>& input, size_t filterHeight, size_t filterWidth, size_t paddingH = 0, size_t paddingW = 0, size_t dilationH = 1, size_t dilationW = 1, bool ceilMode = false);

	template <typename T>
	TensorCore::Tensor<T> MaxPool3D(const TensorCore::Tensor<T>& input, size_t filterDepth, size_t filterHeight, size_t filterWidth,
									size_t paddingD = 0, size_t paddingH = 0, size_t paddingW = 0,
									size_t dilationD = 1, size_t dilationH = 1, size_t dilationW = 1, bool ceilMode = false);
}

#include "pooling.inl"