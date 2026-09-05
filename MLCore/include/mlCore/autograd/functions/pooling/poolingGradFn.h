/// poolingGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class MaxPool1DGradFn : public GradFn<T> {
	public:
		MaxPool1DGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> input, std::shared_ptr<TensorCore::TensorImpl<T>> indices, size_t stride, size_t padding, size_t dilation);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		size_t m_Stride; /// Stride along the length for the pooling operation.
		size_t m_Padding; /// Padding along the length for the pooling operation.
		size_t m_Dilation; /// Dilation along the length for the pooling operation.
	};

	template <typename T>
	class MaxPool2DGradFn : public GradFn<T> {
	public:
		MaxPool2DGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> input, std::shared_ptr<TensorCore::TensorImpl<T>> indices,
						size_t strideH, size_t strideW, size_t paddingH, size_t paddingW, size_t dilationH, size_t dilationW);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		size_t m_StrideH, m_StrideW; /// Stride along height and width dimensions for the pooling operation.

		size_t m_PaddingH, m_PaddingW; /// Padding along height and width dimensions for the pooling operation.

		size_t m_DilationH, m_DilationW; /// Dilation along height and width dimensions for the pooling operation.
	};

	template <typename T>
	class MaxPool3DGradFn : public GradFn<T> {
	public:
		MaxPool3DGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> input, std::shared_ptr<TensorCore::TensorImpl<T>> indices,
						size_t strideD, size_t strideH, size_t strideW,
						size_t paddingD, size_t paddingH, size_t paddingW,
						size_t dilationD, size_t dilationH, size_t dilationW);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		size_t m_StrideD, m_StrideH, m_StrideW; /// Stride along depth, height and width dimensions for the convolution operation.

		size_t m_PaddingD, m_PaddingH, m_PaddingW; /// Padding along depth, height and width dimensions for the convolution operation.

		size_t m_DilationD, m_DilationH, m_DilationW; /// Dilation along depth, height and width dimensions for the convolution operation.
	};
}

#include "poolingGradFn.inl"