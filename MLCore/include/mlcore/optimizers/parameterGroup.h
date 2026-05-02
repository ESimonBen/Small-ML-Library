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

		/*ParameterGroup(std::vector<NN::Parameter<T>>& paramsVec, T learningRate, T weightDecay = static_cast<T>(0))
			: learningRate(learningRate), weightDecay(weightDecay) {
			params.reserve(paramsVec.size());

			for (NN::Parameter<T>& p : paramsVec) {
				params.push_back(p);
			}
		}*/

		ParameterGroup(std::initializer_list<std::reference_wrapper<NN::Parameter<T>>> paramsList, T learningRate, T weightDecay = static_cast<T>(0))
			: params(paramsList), learningRate(learningRate), weightDecay(weightDecay)
		{}
	};
}