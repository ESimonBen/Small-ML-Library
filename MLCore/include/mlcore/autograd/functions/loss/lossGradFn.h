// lossGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class MSEGradFn : public GradFn<T> {
	public:
		MSEGradFn(std::shared_ptr<typename GradFn<T>::Impl> pred, std::shared_ptr<typename GradFn<T>::Impl> target);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		std::shared_ptr<typename GradFn<T>::Impl> targetImpl;
	};

	template <typename T>
	class MAEGradFn : public GradFn<T> {
	public:
		MAEGradFn(std::shared_ptr<typename GradFn<T>::Impl> pred, std::shared_ptr<typename GradFn<T>::Impl> target);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		std::shared_ptr<typename GradFn<T>::Impl> targetImpl;
	};

	template <typename T>
	class BCEGradFn : public GradFn<T> {
	public:
		BCEGradFn(std::shared_ptr<typename GradFn<T>::Impl> pred, std::shared_ptr<typename GradFn<T>::Impl> target);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		std::shared_ptr<typename GradFn<T>::Impl> targetImpl;
	};

	template <typename T>
	class BCEWithLogitsGradFn : public GradFn<T> {
	public:
		BCEWithLogitsGradFn(std::shared_ptr<typename GradFn<T>::Impl> pred, std::shared_ptr<typename GradFn<T>::Impl> target);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		std::shared_ptr<typename GradFn<T>::Impl> targetImpl;
	};

	template <typename T>
	class CEGradFn : public GradFn<T> {
	public:
		CEGradFn(std::shared_ptr<typename GradFn<T>::Impl> pred, std::shared_ptr<typename GradFn<T>::Impl> target);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		std::shared_ptr<typename GradFn<T>::Impl> targetImpl;
	};

	template <typename T>
	class CEWithLogitsGradFn : public GradFn<T> {
	public:
		CEWithLogitsGradFn(std::shared_ptr<typename GradFn<T>::Impl> pred, std::shared_ptr<typename GradFn<T>::Impl> target, size_t axis);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		std::shared_ptr<typename GradFn<T>::Impl> targetImpl;
		size_t axis;
	};
}

#include "lossGradFn.inl"