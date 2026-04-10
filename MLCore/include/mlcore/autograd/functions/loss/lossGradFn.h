// lossGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class MSEGradFn : public GradFn<T> {
	public:
		MSEGradFn(TensorCore::Tensor<T>* pred, TensorCore::Tensor<T>* target);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		TensorCore::Tensor<T>* targetTensor;
	};

	template <typename T>
	class MAEGradFn : public GradFn<T> {
	public:
		MAEGradFn(TensorCore::Tensor<T>* pred, TensorCore::Tensor<T>* target);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		TensorCore::Tensor<T>* targetTensor;
	};

	template <typename T>
	class BCEGradFn : public GradFn<T> {
	public:
		BCEGradFn(TensorCore::Tensor<T>* pred, TensorCore::Tensor<T>* target);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		TensorCore::Tensor<T>* targetTensor;
	};

	template <typename T>
	class CEGradFn : public GradFn<T> {
	public:
		CEGradFn(TensorCore::Tensor<T>* pred, TensorCore::Tensor<T>* target);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		TensorCore::Tensor<T>* targetTensor;
	};
}

#include "lossGradFn.inl"