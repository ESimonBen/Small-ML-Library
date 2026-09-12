/// minPoolingLayers.inl
#include <mlCore/operations/pooling/pooling.h>

namespace MLCore::NN {
	template <typename T>
	inline MinPool1DLayer<T>::MinPool1DLayer(size_t filterLength, size_t stride, size_t padding, size_t dilation, bool ceilMode)
		: m_FilterLength(filterLength), m_Stride(stride), m_Padding(padding), m_Dilation(dilation), m_CeilMode(ceilMode)
	{}

	template <typename T>
	inline TensorCore::Tensor<T> MinPool1DLayer<T>::Forward(const TensorCore::Tensor<T>& input) {
		return Operations::MinPool1D(input, m_FilterLength, m_Stride, m_Padding, m_Dilation, m_CeilMode);
	}

	template <typename T>
	inline MinPool2DLayer<T>::MinPool2DLayer(size_t filterHeight, size_t filterWidth, size_t strideH, size_t strideW,
		size_t paddingH, size_t paddingW, size_t dilationH, size_t dilationW, bool ceilMode)
		: m_FilterHeight(filterHeight), m_FilterWidth(filterWidth), m_StrideH(strideH), m_StrideW(strideW),
		m_PaddingH(paddingH), m_PaddingW(paddingW), m_DilationH(dilationH), m_DilationW(dilationW)
	{}

	template <typename T>
	inline TensorCore::Tensor<T> MinPool2DLayer<T>::Forward(const TensorCore::Tensor<T>& input) {
		return Operations::MinPool2D(input, m_FilterHeight, m_FilterWidth, m_StrideH, m_StrideW, m_PaddingH, m_PaddingW, m_DilationH, m_DilationW, m_CeilMode);
	}

	template <typename T>
	inline MinPool3DLayer<T>::MinPool3DLayer(size_t filterDepth, size_t filterHeight, size_t filterWidth,
		size_t strideD, size_t strideH, size_t strideW,
		size_t paddingD, size_t paddingH, size_t paddingW,
		size_t dilationD, size_t dilationH, size_t dilationW, bool ceilMode)
		: m_FilterDepth(filterDepth), m_FilterHeight(filterHeight), m_FilterWidth(filterWidth),
		m_StrideD(strideD), m_StrideH(strideH), m_StrideW(strideW),
		m_PaddingD(paddingD), m_PaddingH(paddingH), m_PaddingW(paddingW),
		m_DilationD(dilationD), m_DilationH(dilationH), m_DilationW(dilationW), m_CeilMode(ceilMode)
	{}

	template <typename T>
	inline TensorCore::Tensor<T> MinPool3DLayer<T>::Forward(const TensorCore::Tensor<T>& input) {
		return Operations::MinPool3D(input, m_FilterDepth, m_FilterHeight, m_FilterWidth, m_StrideD, m_StrideH, m_StrideW,
									 m_PaddingD, m_PaddingH, m_PaddingW, m_DilationD, m_DilationH, m_DilationW, m_CeilMode);
	}
}