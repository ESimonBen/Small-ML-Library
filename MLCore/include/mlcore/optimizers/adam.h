// adam.h
#pragma once
#include "optimizer.h"
#include <unordered_map>

namespace MLCore::Optimizers {
	template <typename T>
	class Adam : public Optimizer<T> {
	public:
		Adam(std::vector<NN::Parameter<T>>& params, T learningRate = static_cast<T>(.001), T weightDecay = static_cast<T>(0), T beta1 = static_cast<T>(.9),
			 T beta2 = static_cast<T>(.999), T epsilon = static_cast<T>(1e-8));

		Adam(std::vector<ParameterGroup<T>> groups, T beta1 = static_cast<T>(.9), T beta2 = static_cast<T>(.999), T epsilon = static_cast<T>(1e-8));

		virtual void Step() override;

	private:
		T m_Beta1;
		T m_Beta2;
		T m_BetaPow1;
		T m_BetaPow2;
		T m_Epsilon;
		size_t m_Timestep; // Will use this in the future
		std::unordered_map<TensorCore::TensorImpl<T>*, TensorCore::Tensor<T>> m_FirstMoment;
		std::unordered_map<TensorCore::TensorImpl<T>*, TensorCore::Tensor<T>> m_SecondMoment;
	};

	template <typename T>
	class AdamW : public Optimizer<T> {
	public:
		AdamW(std::vector<NN::Parameter<T>>& params, T learningRate = static_cast<T>(.001), T weightDecay = static_cast<T>(0), T beta1 = static_cast<T>(.9),
			T beta2 = static_cast<T>(.999), T epsilon = static_cast<T>(1e-8));

		AdamW(std::vector<ParameterGroup<T>> groups, T beta1 = static_cast<T>(.9), T beta2 = static_cast<T>(.999), T epsilon = static_cast<T>(1e-8));

		virtual void Step() override;

	private:
		T m_Beta1;
		T m_Beta2;
		T m_BetaPow1;
		T m_BetaPow2;
		T m_Epsilon;
		size_t m_Timestep; // Will use this in the future
		std::unordered_map<TensorCore::TensorImpl<T>*, TensorCore::Tensor<T>> m_FirstMoment;
		std::unordered_map<TensorCore::TensorImpl<T>*, TensorCore::Tensor<T>> m_SecondMoment;
	};
}

#include "adam.inl"