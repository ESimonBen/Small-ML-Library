// scalarGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class AddScalarGradFn : public GradFn<T> {
	public:
		AddScalarGradFn(std::shared_ptr<typename GradFn<T>::Impl> a);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;
	};

	template <typename T>
	class SubScalarGradFn : public GradFn<T> {
	public:
		SubScalarGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, bool scalarOnLeft);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		bool scalarOnLeft;
	};

	template <typename T>
	class MulScalarGradFn : public GradFn<T> {
	public:
		MulScalarGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T scalar);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		T scalar;
	};

	template <typename T>
	class DivScalarGradFn : public GradFn<T> {
	public:
		DivScalarGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T scalar, bool scalarOnLeft);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		T scalar;
		bool scalarOnLeft;
	};
}

#include "scalarGradFn.inl"