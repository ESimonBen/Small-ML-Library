// sgd.inl

namespace MLCore::Optimizers {
	template <typename T>
	SGD<T>::SGD(std::vector<std::reference_wrapper<NN::Parameter<T>>> params, T learningRate, T weightDecay)
		: Optimizer<T>(params, learningRate, weightDecay)
	{}

	template <typename T>
	SGD<T>::SGD(std::vector<NN::Parameter<T>>& params, T learningRate, T weightDecay)
		: Optimizer<T>(params, learningRate, weightDecay)
	{}

	template <typename T>
	SGD<T>::SGD(std::vector<ParameterGroup<T>> groups)
		: Optimizer<T>(groups)
	{}

	template <typename T>
	void SGD<T>::Step() {
		this->ClipGradients();

		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			T learningRate = paramGroup.learningRate;
			T weightDecay = paramGroup.weightDecay;

			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				TensorCore::Tensor<T>& param = p.Data();

				if (!param.RequiresGrad() || !param.HasGrad()) {
					continue;
				}

				TensorCore::Tensor<T> grad = param.Grad();
				size_t size = param.NumElements();

				for (size_t i = 0; i < size; ++i) {
					T gradScalar = grad[i];

					if (weightDecay != static_cast<T>(0)) {
						gradScalar += weightDecay * param[i];
					}

					param[i] -= learningRate * gradScalar;
				}
			}
		}
	}

	template <typename T>
	SGDMomentum<T>::SGDMomentum(std::vector<std::reference_wrapper<NN::Parameter<T>>> params, T learningRate, T momentum, T weightDecay, T dampening, bool nesterov)
		: Optimizer<T>(params, learningRate, weightDecay), m_Momentum(momentum), m_Dampening(dampening), m_Nesterov(nesterov) {
			for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
				for (auto& ref : paramGroup.params) {
					NN::Parameter<T>& p = ref.get();
					NN::ParamID paramID = p.id;
					TensorCore::Tensor<T>& param = p.Data();

					TensorCore::Tensor<T> velocity{ param.GetShape(), param.GetAllocator() };
					velocity.Fill(static_cast<T>(0));
					m_Velocities.try_emplace(paramID, velocity);
				}
			}
	}

	template <typename T>
	SGDMomentum<T>::SGDMomentum(std::vector<NN::Parameter<T>>& params, T learningRate, T momentum, T weightDecay, T dampening, bool nesterov)
		: Optimizer<T>(params, learningRate, weightDecay), m_Momentum(momentum), m_Dampening(dampening), m_Nesterov(nesterov) {
			for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
				for (auto& ref : paramGroup.params) {
					NN::Parameter<T>& p = ref.get();
					NN::ParamID paramID = p.id;
					TensorCore::Tensor<T>& param = p.Data();

					TensorCore::Tensor<T> velocity{ param.GetShape(), param.GetAllocator() };
					velocity.Fill(static_cast<T>(0));
					m_Velocities.try_emplace(paramID, velocity);
				}
			}
	}

	template <typename T>
	SGDMomentum<T>::SGDMomentum(std::vector<ParameterGroup<T>> groups, T momentum, T dampening, bool nesterov)
		: Optimizer<T>(groups), m_Momentum(momentum), m_Dampening(dampening), m_Nesterov(nesterov) {
		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				NN::ParamID paramID = p.id;
				TensorCore::Tensor<T>& param = p.Data();

				TensorCore::Tensor<T> velocity{ param.GetShape(), param.GetAllocator() };
				velocity.Fill(static_cast<T>(0));
				m_Velocities.try_emplace(paramID, velocity);
			}
		}
	}

	template <typename T>
	void SGDMomentum<T>::Step() {
		this->ClipGradients();

		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			T learningRate = paramGroup.learningRate;
			T weightDecay = paramGroup.weightDecay;

			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				NN::ParamID paramID = p.id;
				TensorCore::Tensor<T>& param = p.Data();

				auto velocityIt = m_Velocities.find(paramID);

				if (velocityIt == m_Velocities.end()) {
					throw std::runtime_error("ERROR: Step: Missing optimizer state");
				}

				TensorCore::Tensor<T>& velocity = velocityIt->second;

				if (!param.RequiresGrad() || !param.HasGrad()) {
					continue;
				}

				TensorCore::Tensor<T> grad = param.Grad();

				size_t size = param.NumElements();
				for (size_t j = 0; j < size; ++j) {
					T gradScalar = grad[j];

					// Weight decay
					if (weightDecay != static_cast<T>(0)) {
						gradScalar += weightDecay * param[j];
					}

					T scaledGrad = (static_cast<T>(1) - m_Dampening) * gradScalar;

					// Momentum with dampening
					velocity[j] = (m_Momentum * velocity[j]) + scaledGrad;

					// Nesterov or standard stochastic gradient descent
					if (m_Nesterov) {
						param[j] -= learningRate * (scaledGrad + m_Momentum * velocity[j]);
					}
					else {
						param[j] -= learningRate * velocity[j];
					}
				}
			}
		}
	}
}