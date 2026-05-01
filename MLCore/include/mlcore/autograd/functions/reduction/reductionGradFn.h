// reductionGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class SumGradFn : public GradFn<T> {
	public:
		SumGradFn(std::shared_ptr<typename GradFn<T>::Impl> a);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator);

	private:
		Utils::Shape inputShape;
	};

	template <typename T>
	class MaxGradFn : public GradFn<T> {
	public:
		MaxGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T maxValue);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		Utils::Shape inputShape;
		T maxValue;
	};

	template <typename T>
	class MinGradFn : public GradFn<T> {
	public:
		MinGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T minValue);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		Utils::Shape inputShape;
		T minValue;
	};

	template <typename T>
	class AxisSumGradFn : public GradFn<T> {
	public:
		AxisSumGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis, bool keepDims);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		Utils::Shape inputShape;
		size_t m_Axis;
		bool m_KeepDims;
	};

	template <typename T>
	class AxisMaxGradFn : public GradFn<T> {
	public:
		AxisMaxGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis, bool keepDims);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		size_t m_Axis;
		bool m_KeepDims;
	};

	template <typename T>
	class AxisMinGradFn : public GradFn<T> {
	public:
		AxisMinGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis, bool keepDims);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) override;

	private:
		size_t m_Axis;
		bool m_KeepDims;
	};
}

#include "reductionGradFn.inl"