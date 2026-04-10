// elementwiseGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/autograd/gradientUtils.h>
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class AddGradFn : public GradFn<T> {
	public:
		AddGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;
	};

	template <typename T>
	class SubGradFn : public GradFn<T> {
	public:
		SubGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput);
	};

	template <typename T>
	class MulGradFn : public GradFn<T> {
	public:
		MulGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

		// Maybe store the saved forward values of the tensors (needed for multiply and divide)
	};

	template <typename T>
	class DivGradFn : public GradFn<T> {
	public:
		DivGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
			: GradFn<T>({ a, b })
		{
		}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

		// Maybe store the saved forward values of the tensors (needed for multiply and divide)
	};

	template <typename T>
	class PowerGradFn : public GradFn<T> {
	public:
		PowerGradFn(TensorCore::Tensor<T>* a, T exponent);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		T exponent;
	};
}

#include "elementwiseGradFn.inl"