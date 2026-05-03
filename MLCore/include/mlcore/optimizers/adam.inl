// adam.inl
#include <cmath>

namespace MLCore::Optimizers {
	template <typename T>
	Adam<T>::Adam(std::vector <std::reference_wrapper<NN::Parameter<T>>>& params, T learningRate, T weightDecay, T beta1, T beta2, T epsilon)
		: Optimizer<T>(params, learningRate, weightDecay), m_Beta1(beta1), m_Beta2(beta2), m_BetaPow1(static_cast<T>(1)), m_BetaPow2(static_cast<T>(1)),
		m_Epsilon(epsilon), m_Timestep(0) {
			for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
				for (auto& ref : paramGroup.params) {
					NN::Parameter<T>& p = ref.get();
					NN::ParamID paramID = p.id;
					TensorCore::Tensor<T>& param = p.Data();

					TensorCore::Tensor<T> m{ param.GetShape(), param.GetAllocator() };
					TensorCore::Tensor<T> v{ param.GetShape(), param.GetAllocator() };

					m.Fill(static_cast<T>(0));
					v.Fill(static_cast<T>(0));

					m_FirstMoment.try_emplace(paramID, m);
					m_SecondMoment.try_emplace(paramID, v);
				}
			}
	}

	template <typename T>
	Adam<T>::Adam(std::vector<NN::Parameter<T>>& params, T learningRate, T weightDecay, T beta1, T beta2, T epsilon)
		: Optimizer<T>(params, learningRate, weightDecay), m_Beta1(beta1), m_Beta2(beta2), m_BetaPow1(static_cast<T>(1)), m_BetaPow2(static_cast<T>(1)),
		  m_Epsilon(epsilon), m_Timestep(0) {
			  for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
				  for (auto& ref : paramGroup.params) {
					  NN::Parameter<T>& p = ref.get();
					  NN::ParamID paramID = p.id;
					  TensorCore::Tensor<T>& param = p.Data();

					  TensorCore::Tensor<T> m{ param.GetShape(), param.GetAllocator() };
					  TensorCore::Tensor<T> v{ param.GetShape(), param.GetAllocator() };

					  m.Fill(static_cast<T>(0));
					  v.Fill(static_cast<T>(0));

					  m_FirstMoment.try_emplace(paramID, m);
					  m_SecondMoment.try_emplace(paramID, v);
				  }
			  }
	}

	template <typename T>
	Adam<T>::Adam(std::vector<ParameterGroup<T>> groups, T beta1, T beta2, T epsilon)
		: Optimizer<T>(groups), m_Beta1(beta1), m_Beta2(beta2), m_BetaPow1(static_cast<T>(1)), m_BetaPow2(static_cast<T>(1)),
		m_Epsilon(epsilon), m_Timestep(0) {
		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				NN::ParamID paramID = p.id;
				TensorCore::Tensor<T>& param = p.Data();

				TensorCore::Tensor<T> m{ param.GetShape(), param.GetAllocator() };
				TensorCore::Tensor<T> v{ param.GetShape(), param.GetAllocator() };

				m.Fill(static_cast<T>(0));
				v.Fill(static_cast<T>(0));

				m_FirstMoment.try_emplace(paramID, m);
				m_SecondMoment.try_emplace(paramID, v);
			}
		}
	}

	template <typename T>
	void Adam<T>::Step() {
		m_Timestep++;
		this->ClipGradients();

		m_BetaPow1 *= m_Beta1;
		m_BetaPow2 *= m_Beta2;
		T bias1 = (static_cast<T>(1) - m_BetaPow1);
		T bias2 = (static_cast<T>(1) - m_BetaPow2);

		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			T learningRate = paramGroup.learningRate;
			T weightDecay = paramGroup.weightDecay;

			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				NN::ParamID paramID = p.id;
				TensorCore::Tensor<T>& param = p.Data();

				auto mIt = m_FirstMoment.find(paramID);
				auto vIt = m_SecondMoment.find(paramID);

				if (mIt == m_FirstMoment.end() || vIt == m_SecondMoment.end()) {
					throw std::runtime_error("ERROR: Step: Missing optimizer state");
				}

				TensorCore::Tensor<T>& m = mIt->second;
				TensorCore::Tensor<T>& v = vIt->second;


				if (!param.RequiresGrad() || !param.HasGrad()) {
					continue;
				}

				if (m.NumElements() == 0) {
					m = TensorCore::Tensor<T>{ param.GetShape(), param.GetAllocator() };
					m.Fill(static_cast<T>(0));
				}

				if (v.NumElements() == 0) {
					v = TensorCore::Tensor<T>{ param.GetShape(), param.GetAllocator() };
					v.Fill(static_cast<T>(0));
				}

				TensorCore::Tensor<T> grad = param.Grad();

				size_t size = param.NumElements();

				for (size_t i = 0; i < size; ++i) {
					T gradScalar = grad[i];

					// Weight decay
					if (weightDecay != static_cast<T>(0)) {
						gradScalar += weightDecay * param[i];
					}

					// Update biased moments
					m[i] = m_Beta1 * m[i] + (static_cast<T>(1) - m_Beta1) * gradScalar;
					v[i] = m_Beta2 * v[i] + (static_cast<T>(1) - m_Beta2) * (gradScalar * gradScalar);

					// Bias correction
					T m_hat = m[i] / bias1;
					T v_hat = v[i] / bias2;

					param[i] -= learningRate * (m_hat / (std::sqrt(v_hat) + m_Epsilon));
				}
			}
		}
	}

	template <typename T>
	AdamW<T>::AdamW(std::vector <std::reference_wrapper<NN::Parameter<T>>>& params, T learningRate, T weightDecay, T beta1, T beta2, T epsilon)
		: Optimizer<T>(params, learningRate, weightDecay), m_Beta1(beta1), m_Beta2(beta2), m_BetaPow1(static_cast<T>(1)), m_BetaPow2(static_cast<T>(1)),
		m_Epsilon(epsilon), m_Timestep(0) {
		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				NN::ParamID paramID = p.id;
				TensorCore::Tensor<T>& param = p.Data();

				TensorCore::Tensor<T> m{ param.GetShape(), param.GetAllocator() };
				TensorCore::Tensor<T> v{ param.GetShape(), param.GetAllocator() };

				m.Fill(static_cast<T>(0));
				v.Fill(static_cast<T>(0));

				m_FirstMoment.try_emplace(paramID, m);
				m_SecondMoment.try_emplace(paramID, v);
			}
		}
	}

	template <typename T>
	AdamW<T>::AdamW(std::vector<NN::Parameter<T>>& params, T learningRate, T weightDecay, T beta1, T beta2, T epsilon)
		: Optimizer<T>(params, learningRate, weightDecay), m_Beta1(beta1), m_Beta2(beta2), m_BetaPow1(static_cast<T>(1)), m_BetaPow2(static_cast<T>(1)),
		  m_Epsilon(epsilon), m_Timestep(0) {
			for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
				for (auto& ref : paramGroup.params) {
					NN::Parameter<T>& p = ref.get();
					NN::ParamID paramID = p.id;
					TensorCore::Tensor<T>& param = p.Data();

					TensorCore::Tensor<T> m{ param.GetShape(), param.GetAllocator() };
					TensorCore::Tensor<T> v{ param.GetShape(), param.GetAllocator() };

					m.Fill(static_cast<T>(0));
					v.Fill(static_cast<T>(0));

					m_FirstMoment.try_emplace(paramID, m);
					m_SecondMoment.try_emplace(paramID, v);
				}
			}
	}

	template <typename T>
	AdamW<T>::AdamW(std::vector<ParameterGroup<T>> groups, T beta1, T beta2, T epsilon)
		: Optimizer<T>(groups), m_Beta1(beta1), m_Beta2(beta2), m_BetaPow1(static_cast<T>(1)), m_BetaPow2(static_cast<T>(1)),
		m_Epsilon(epsilon), m_Timestep(0) {
		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				NN::ParamID paramID = p.id;
				TensorCore::Tensor<T>& param = p.Data();

				TensorCore::Tensor<T> m{ param.GetShape(), param.GetAllocator() };
				TensorCore::Tensor<T> v{ param.GetShape(), param.GetAllocator() };

				m.Fill(static_cast<T>(0));
				v.Fill(static_cast<T>(0));

				m_FirstMoment.try_emplace(paramID, m);
				m_SecondMoment.try_emplace(paramID, v);
			}
		}
	}

	template <typename T>
	void AdamW<T>::Step() {
		m_Timestep++;
		this->ClipGradients();

		m_BetaPow1 *= m_Beta1;
		m_BetaPow2 *= m_Beta2;
		T bias1 = (static_cast<T>(1) - m_BetaPow1);
		T bias2 = (static_cast<T>(1) - m_BetaPow2);

		for (ParameterGroup<T>& paramGroup : this->m_ParamGroups) {
			T learningRate = paramGroup.learningRate;
			T weightDecay = paramGroup.weightDecay;

			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				NN::ParamID paramID = p.id;
				TensorCore::Tensor<T>& param = p.Data();

				auto mIt = m_FirstMoment.find(paramID);
				auto vIt = m_SecondMoment.find(paramID);

				if (mIt == m_FirstMoment.end() || vIt == m_SecondMoment.end()) {
					throw std::runtime_error("ERROR: Step: Missing optimizer state");
				}

				TensorCore::Tensor<T>& m = mIt->second;
				TensorCore::Tensor<T>& v = vIt->second;


				if (!param.RequiresGrad() || !param.HasGrad()) {
					continue;
				}

				if (m.NumElements() == 0) {
					m = TensorCore::Tensor<T>{ param.GetShape(), param.GetAllocator() };
					m.Fill(static_cast<T>(0));
				}

				if (v.NumElements() == 0) {
					v = TensorCore::Tensor<T>{ param.GetShape(), param.GetAllocator() };
					v.Fill(static_cast<T>(0));
				}

				TensorCore::Tensor<T> grad = param.Grad();

				size_t size = param.NumElements();

				for (size_t i = 0; i < size; ++i) {
					T gradScalar = grad[i];

					// Weight decay
					if (weightDecay != static_cast<T>(0)) {
						param[i] -= learningRate * weightDecay * param[i];
					}

					// Update biased moments
					m[i] = m_Beta1 * m[i] + (static_cast<T>(1) - m_Beta1) * gradScalar;
					v[i] = m_Beta2 * v[i] + (static_cast<T>(1) - m_Beta2) * (gradScalar * gradScalar);

					// Bias correction
					T m_hat = m[i] / bias1;
					T v_hat = v[i] / bias2;

					param[i] -= learningRate * (m_hat / (std::sqrt(v_hat) + m_Epsilon));
				}
			}
		}
	}
}