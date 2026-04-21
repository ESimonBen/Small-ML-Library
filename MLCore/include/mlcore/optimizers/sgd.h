// sgd.h
#pragma once
#include "optimizer.h"

namespace MLCore::Optimizers {
	// Stochastic Gradient Descent (Regular)
	template <typename T>
	class SGD : public Optimizer<T> {
	public:
		SGD(std::vector<Parameter<T>>& params, T learningRate, T weightDecay = static_cast<T>(0));

		virtual void Step() override;

	private:
		T m_WeightDecay;
	};

	// Stochastic Gradient Descent (With Momentum)
	template <typename T>
	class SGDMomentum : public Optimizer<T> {
	public:
		SGDMomentum(std::vector<Parameter<T>>& params, T learningRate, T momentum, T weightDecay = static_cast<T>(0), T m_Dampening = static_cast<T>(0), bool nesterov = false);

		virtual void Step() override;

	private:
		T m_Momentum;
		T m_WeightDecay;
		T m_Dampening;
		bool m_Nesterov;
		std::vector<TensorCore::Tensor<T>> m_Velocities;
	};
}

#include "sgd.inl"