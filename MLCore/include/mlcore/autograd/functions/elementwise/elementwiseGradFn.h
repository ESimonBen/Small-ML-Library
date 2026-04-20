// elementwiseGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class AddGradFn : public GradFn<T> {
	public:
		AddGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;
	};

	template <typename T>
	class SubGradFn : public GradFn<T> {
	public:
		SubGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput);
	};

	template <typename T>
	class MulGradFn : public GradFn<T> {
	public:
		MulGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

		// Maybe store the saved forward values of the tensors (needed for multiply and divide)
	};

	template <typename T>
	class DivGradFn : public GradFn<T> {
	public:
		DivGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

		// Maybe store the saved forward values of the tensors (needed for multiply and divide)
	};

	template <typename T>
	class PowerGradFn : public GradFn<T> {
	public:
		PowerGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T exponent);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override;

	private:
		T exponent;
	};
}

#include "elementwiseGradFn.inl"