// sgd.h
#pragma once
#include "optimizer.h"
#include <unordered_map>

namespace MLCore::Optimizers {
	// Stochastic Gradient Descent (Regular)
	template <typename T>
	class SGD : public Optimizer<T> {
	public:
		SGD(std::vector<std::reference_wrapper<NN::Parameter<T>>> params, T learningRate, T weightDecay = static_cast<T>(0));
		SGD(std::vector<NN::Parameter<T>>& params, T learningRate, T weightDecay = static_cast<T>(0));
		SGD(std::vector<ParameterGroup<T>> groups);

		virtual void Step() override;

		virtual std::string TypeName() const override;

		// NOTE:
		// Optimizer parameter groups must be reconstructed
		// in the same order before loading state.

		virtual void SaveState(Serialization::BinaryWriter& writer, const NN::Module<T>& model) const override;
		virtual void LoadState(Serialization::BinaryReader& reader, NN::Module<T>& model) override;
	};

	// Stochastic Gradient Descent (With Momentum)
	template <typename T>
	class SGDMomentum : public Optimizer<T> {
	public:
		SGDMomentum(std::vector<std::reference_wrapper<NN::Parameter<T>>> params, T learningRate, T momentum, T weightDecay = static_cast<T>(0), T dampening = static_cast<T>(0), bool nesterov = false);
		SGDMomentum(std::vector<NN::Parameter<T>>& params, T learningRate, T momentum, T weightDecay = static_cast<T>(0), T dampening = static_cast<T>(0), bool nesterov = false);
		SGDMomentum(std::vector<ParameterGroup<T>> groups, T momentum, T dampening = static_cast<T>(0), bool nesterov = false);

		virtual void Step() override;

		virtual std::string TypeName() const override;

		// NOTE:
		// Optimizer parameter groups must be reconstructed
		// in the same order before loading state.

		virtual void SaveState(Serialization::BinaryWriter& writer, const NN::Module<T>& model) const override;
		virtual void LoadState(Serialization::BinaryReader& reader, NN::Module<T>& model) override;

	private:
		T m_Momentum;
		T m_Dampening;
		bool m_Nesterov;
		std::unordered_map<NN::ParamID, TensorCore::Tensor<T>> m_Velocities;
	};
}

#include "sgd.inl"