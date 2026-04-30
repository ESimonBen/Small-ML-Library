// elementwiseGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class AddGradFn : public GradFn<T> {
	public:
		AddGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;
	};

	template <typename T>
	class SubGradFn : public GradFn<T> {
	public:
		SubGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator);
	};

	template <typename T>
	class MulGradFn : public GradFn<T> {
	public:
		MulGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

		// Maybe store the saved forward values of the tensors (needed for multiply and divide)
	};

	template <typename T>
	class DivGradFn : public GradFn<T> {
	public:
		DivGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

		// Maybe store the saved forward values of the tensors (needed for multiply and divide)
	};

	template <typename T>
	class PowerGradFn : public GradFn<T> {
	public:
		PowerGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T exponent);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		T exponent;
	};

	template <typename T>
	class AbsGradFn : public GradFn<T> {
	public:
		AbsGradFn(std::shared_ptr<typename GradFn<T>::Impl> input);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	};

	template <typename T>
	class ClampGradFn : public GradFn<T> {
	public:
		ClampGradFn(std::shared_ptr<typename GradFn<T>::Impl> input, T min, T max);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		T m_Min;
		T m_Max;
	};

	template <typename T>
	class LogGradFn : public GradFn<T> {
	public:
		LogGradFn(std::shared_ptr<typename GradFn<T>::Impl> input);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	};

	template <typename T>
	class ExpGradFn : public GradFn<T> {
	public:
		ExpGradFn(std::shared_ptr<typename GradFn<T>::Impl> input);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	};
}

#include "elementwiseGradFn.inl"