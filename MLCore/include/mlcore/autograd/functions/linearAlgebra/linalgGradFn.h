// linalgGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/linearAlgebra/linalg.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class DotGradFn : public GradFn<T> {
	public:
		DotGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;
	};

	template <typename T>
	class MatMulGradFn : public GradFn<T> {
	public:
		MatMulGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;
	};

	template <typename T>
	class TransposeGradFn : public GradFn<T> {
	public:
		TransposeGradFn(TensorCore::Tensor<T>* a);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;
	};
}

#include "linalgGradFn.inl"