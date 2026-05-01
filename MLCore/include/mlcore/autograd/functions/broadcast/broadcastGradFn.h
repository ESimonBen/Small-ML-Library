// broadcastGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class SqueezeGradFn : public GradFn<T> {
	public:
		SqueezeGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		size_t m_Axis;
	};

	template <typename T>
	class UnsqueezeGradFn : public GradFn<T> {
	public:
		UnsqueezeGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		size_t m_Axis;
	};
}

#include "broadcastGradFn.inl"