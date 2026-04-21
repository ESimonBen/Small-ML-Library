// activationGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class ReLUGradFn : public GradFn<T> {
	public:
		ReLUGradFn(std::shared_ptr<typename GradFn<T>::Impl> a);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;
	};

	template <typename T>
	class LeakyReLUGradFn : public GradFn<T> {
	public:
		LeakyReLUGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T alpha);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		T alpha;
	};

	template <typename T>
	class SigmoidGradFn : public GradFn<T> {
	public:
		SigmoidGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		std::shared_ptr<typename GradFn<T>::Impl> outputImpl;
	};

	template <typename T>
	class TanhGradFn : public GradFn<T> {
	public:
		TanhGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		std::shared_ptr<typename GradFn<T>::Impl> outputImpl;
	};

	template <typename T>
	class SoftmaxGradFn : public GradFn<T> {
	public:
		SoftmaxGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		std::shared_ptr<typename GradFn<T>::Impl> outputImpl;
	};

	template <typename T>
	class AxisSoftmaxGradFn : public GradFn<T> {
	public:
		AxisSoftmaxGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b, size_t axis);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		std::shared_ptr<typename GradFn<T>::Impl> outputImpl;
		size_t axis;
	};
}

#include "activationGradFn.inl"