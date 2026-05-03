// parameterGroup.h
#pragma once
#include <functional>
#include <mlCore/parameters/parameter.h>

namespace MLCore::Optimizers {
	template <typename T>
	struct ParameterGroup {
		std::vector<std::reference_wrapper<NN::Parameter<T>>> params; // I feel like this will cause problems soon

		T learningRate;
		T weightDecay;

		ParameterGroup(std::initializer_list<std::reference_wrapper<NN::Parameter<T>>> paramsList, T learningRate, T weightDecay = static_cast<T>(0))
			: params(paramsList), learningRate(learningRate), weightDecay(weightDecay)
		{}

		ParameterGroup(std::vector<std::reference_wrapper<NN::Parameter<T>>> paramsList, T learningRate, T weightDecay = static_cast<T>(0))
			: params(std::move(paramsList)), learningRate(learningRate), weightDecay(weightDecay)
		{}
	};
}