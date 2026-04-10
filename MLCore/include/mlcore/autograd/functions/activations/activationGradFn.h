// activationGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class ReLUGradFn : public GradFn<T> {
	public:
		ReLUGradFn(TensorCore::Tensor<T>* a);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;
	};

	template <typename T>
	class LeakyReLUGradFn : public GradFn<T> {
	public:
		LeakyReLUGradFn(TensorCore::Tensor<T>* a, T alpha);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		T alpha;
	};

	template <typename T>
	class SigmoidGradFn : public GradFn<T> {
	public:
		SigmoidGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		TensorCore::Tensor<T>* outputTensor;
	};

	template <typename T>
	class TanhGradFn : public GradFn<T> {
	public:
		TanhGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		TensorCore::Tensor<T>* outputTensor;
	};

	template <typename T>
	class SoftmaxGradFn : public GradFn<T> {
	public:
		SoftmaxGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		TensorCore::Tensor<T>* outputTensor;
	};
}

#include "activationGradFn.inl"