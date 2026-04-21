// sgd.inl

namespace MLCore::Optimizers {
	template <typename T>
	SGD<T>::SGD(std::vector<Parameter<T>>& params, T learningRate, T weightDecay)
		: Optimizer<T>(params, learningRate), m_WeightDecay(weightDecay)
	{}

	template <typename T>
	void SGD<T>::Step() {
		this->ClipGradients();

		for (Parameter<T>& p : this->m_Params) {
			TensorCore::Tensor<T>& param = p.Data();

			if (!param.RequiresGrad() || !param.HasGrad()) {
				continue;
			}

			TensorCore::Tensor<T> grad = param.Grad();
			size_t size = param.NumElements();

			for (size_t i = 0; i < size; ++i) {
				T gradScalar = grad[i];

				if (m_WeightDecay != static_cast<T>(0)) {
					gradScalar += m_WeightDecay * param[i];
				}

				param[i] -= this->m_LearningRate * gradScalar;
			}
		}
	}

	template <typename T>
	SGDMomentum<T>::SGDMomentum(std::vector<Parameter<T>>& params, T learningRate, T momentum, T weightDecay, T dampening, bool nesterov)
		: Optimizer<T>(params, learningRate), m_Momentum(momentum), m_WeightDecay(weightDecay), m_Dampening(dampening), m_Nesterov(nesterov) {
		for (Parameter<T>& p : this->m_Params) {
			TensorCore::Tensor<T>& param = p.Data();
			TensorCore::Tensor<T> velocity{ param.GetShape(), param.GetAllocator() };
			velocity.Fill(static_cast<T>(0));
			m_Velocities.push_back(velocity);
		}
	}

	template <typename T>
	void SGDMomentum<T>::Step() {
		this->ClipGradients();
		size_t sizeParams = this->m_Params.size();

		for (size_t i = 0; i < sizeParams; ++i) {
			TensorCore::Tensor<T>& param = this->m_Params[i].Data();

			if (!param.RequiresGrad() || !param.HasGrad()) {
				continue;
			}

			TensorCore::Tensor<T> grad = param.Grad();
			TensorCore::Tensor<T>& velocity = m_Velocities[i];

			size_t size = param.NumElements();
			for (size_t j = 0; j < size; ++j) {
				T gradScalar = grad[j];

				// Weight decay
				if (m_WeightDecay != static_cast<T>(0)) {
					gradScalar += m_WeightDecay * param[j];
				}

				// Momentum with dampening
				velocity[j] = (m_Momentum * velocity[j]) + ((static_cast<T>(1) - m_Dampening) * gradScalar);

				// Nesterov or standard stochastic gradient descent
				if (m_Nesterov) {
					param[j] -= this->m_LearningRate * (m_Momentum * velocity[j] + gradScalar);
				}
				else {
					param[j] -= this->m_LearningRate * velocity[j];
				}
			}
		}
	}
}