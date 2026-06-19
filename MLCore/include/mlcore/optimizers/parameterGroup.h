 /// parameterGroup.h
#pragma once
#include <functional>
#include <mlCore/parameters/parameter.h>

namespace MLCore::Optimizers {
	/// <summary>
	/// Holds a group of model parameters together with optimizer hyperparameters used when updating them.
	/// </summary>
	/// <typeparam name="T">The numeric type used for the group's hyperparameters and underlying parameters (e.g., float or double).</typeparam>
	template <typename T>
	struct ParameterGroup {
		std::vector<std::reference_wrapper<NN::Parameter<T>>> params; /// A container holding non-owning references to parameters.

		T learningRate; /// The learning rate value of this group of parameters
		T weightDecay; /// The weight decay value of this group of parameters

		/// <summary>
		/// Constructs a ParameterGroup that holds references to network parameters along with a learning rate and an optional weight decay.
		/// </summary>
		/// <param name="paramsList">Initializer list of reference_wrappers to NN::Parameter<T> objects to include in the group; parameters are stored by reference.</param>
		/// <param name="learningRate">Learning rate to apply to all parameters in the group.</param>
		/// <param name="weightDecay">Optional weight decay (regularization) value applied during updates; defaults to 0.</param>
		ParameterGroup(std::initializer_list<std::reference_wrapper<NN::Parameter<T>>> paramsList, T learningRate, T weightDecay = static_cast<T>(0))
			: params(paramsList), learningRate(learningRate), weightDecay(weightDecay)
		{}

		/// <summary>
		/// Constructs a ParameterGroup that stores references to parameters and sets the learning rate and optional weight decay.
		/// </summary>
		/// <param name="paramsList">A vector of std::reference_wrapper<NN::Parameter<T>> objects. The vector is moved into the object's params member.</param>
		/// <param name="learningRate">The learning rate to apply to this parameter group.</param>
		/// <param name="weightDecay">Optional weight decay (L2 regularization) factor; defaults to zero.</param>
		ParameterGroup(std::vector<std::reference_wrapper<NN::Parameter<T>>> paramsList, T learningRate, T weightDecay = static_cast<T>(0))
			: params(std::move(paramsList)), learningRate(learningRate), weightDecay(weightDecay)
		{}
	};
}