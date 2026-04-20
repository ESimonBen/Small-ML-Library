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
		DotGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;
	};

	template <typename T>
	class MatMulGradFn : public GradFn<T> {
	public:
		MatMulGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;
	};

	template <typename T>
	class TransposeGradFn : public GradFn<T> {
	public:
		TransposeGradFn(std::shared_ptr<typename GradFn<T>::Impl> a);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;
	};
}

#include "linalgGradFn.inl"