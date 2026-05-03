// optimizer.h
#pragma once
#include <vector>
#include <functional>
#include <mlCore/tensor/tensor.h>
#include <mlCore/optimizers/parameterGroup.h>

namespace MLCore::Optimizers {
	template <typename T>
	class Optimizer {
	public:
		Optimizer(std::vector<std::reference_wrapper<NN::Parameter<T>>> params, T learningRate, T weightDecay = static_cast<T>(0));
		Optimizer(std::vector<NN::Parameter<T>>& params, T learningRate, T weightDecay = static_cast<T>(0));
		Optimizer(std::vector<ParameterGroup<T>> groups);

		virtual ~Optimizer() = default;

		// Update rule (changes with different optimizers)
		virtual void Step() = 0;
		virtual void ZeroGrad();

		std::vector<ParameterGroup<T>>& ParamGroups();
		void SetClipGradNorm(T maxNorm);

	protected:
		// Functions
		void ClipGradients();
		
		// Members
		std::vector<ParameterGroup<T>> m_ParamGroups;

	private:
		bool m_UseClip = false;
		T m_MaxNorm = static_cast<T>(0);
	};
}

#include "optimizer.inl"