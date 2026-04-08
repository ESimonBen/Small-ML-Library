// scalarGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class AddScalarGradFn : public GradFn<T> {
	public:
		AddScalarGradFn(TensorCore::Tensor<T>* a);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;
	};

	template <typename T>
	class SubScalarGradFn : public GradFn<T> {
	public:
		SubScalarGradFn(TensorCore::Tensor<T>* a, bool scalarOnLeft);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		bool scalarOnLeft;
	};

	template <typename T>
	class MulScalarGradFn : public GradFn<T> {
	public:
		MulScalarGradFn(TensorCore::Tensor<T>* a, T scalar);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		T scalar;
	};

	template <typename T>
	class DivScalarGradFn : public GradFn<T> {
	public:
		DivScalarGradFn(TensorCore::Tensor<T>* a, T scalar, bool scalarOnLeft);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		T scalar;
		bool scalarOnLeft;
	};
}

#include "scalarGradFn.inl"