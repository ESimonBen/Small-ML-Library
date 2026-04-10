// reductionGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class SumGradFn : public GradFn<T> {
	public:
		SumGradFn(TensorCore::Tensor<T>* a);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput);

	private:
		Utils::Shape inputShape;
	};

	// Might remove this later
	template <typename T>
	class MeanGradFn : public GradFn<T> {
	public:
		MeanGradFn(TensorCore::Tensor<T>* a);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		Utils::Shape inputShape;
		size_t numElements;
	};

	template <typename T>
	class MaxGradFn : public GradFn<T> {
	public:
		MaxGradFn(TensorCore::Tensor<T>* a, T maxValue);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		Utils::Shape inputShape;
		T maxValue;
	};

	template <typename T>
	class MinGradFn : public GradFn<T> {
	public:
		MinGradFn(TensorCore::Tensor<T>* a, T minValue);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		Utils::Shape inputShape;
		T minValue;
	};

	template <typename T>
	class AxisSumGradFn : public GradFn<T> {
	public:
		AxisSumGradFn(TensorCore::Tensor<T>* a, size_t axis);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		size_t axis;
		Utils::Shape inputShape;
	};
}

#include "reductionGradFn.inl"