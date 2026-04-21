// adam.inl
#include <cmath>

namespace MLCore::Optimizers {
	template <typename T>
	Adam<T>::Adam(std::vector<Parameter<T>>& params, T learningRate, T beta1, T beta2, T epsilon, T weightDecay)
		: Optimizer<T>(params), m_LearningRate(learningRate), m_Beta1(beta1), m_Beta2(beta2), m_BetaPow1(beta1), m_BetaPow2(beta2),
		  m_Epsilon(epsilon), m_WeightDecay(weightDecay), m_Timestep(0) {
		for (Parameter<T>& p : this->m_Params) {
			TensorCore::Tensor<T>& param = p.Data();

			TensorCore::Tensor<T> m{ param.GetShape(), param.GetAllocator() };
			TensorCore::Tensor<T> v{ param.GetShape(), param.GetAllocator() };

			m.Fill(static_cast<T>(0));
			v.Fill(static_cast<T>(0));

			m_FirstMoment.push_back(m);
			m_SecondMoment.push_back(v);
		}
	}

	template <typename T>
	void Adam<T>::Step() {
		m_Timestep++;

		m_BetaPow1 *= m_Beta1;
		m_BetaPow2 *= m_Beta2;
		T bias1 = (static_cast<T>(1) - m_BetaPow1);
		T bias2 = (static_cast<T>(1) - m_BetaPow2);

		size_t sizeParams = this->m_Params.size();

		for (size_t i = 0; i < sizeParams; ++i) {
			TensorCore::Tensor<T>& param = this->m_Params[i].Data();

			if (!param.RequiresGrad() || !param.HasGrad()) {
				continue;
			}

			TensorCore::Tensor<T> grad = param.Grad();
			TensorCore::Tensor<T>& m = m_FirstMoment[i];
			TensorCore::Tensor<T>& v = m_SecondMoment[i];

			size_t size = param.NumElements();

			for (size_t j = 0; j < size; ++j) {
				T gradScalar = grad[j];

				// Weight decay
				if (m_WeightDecay != static_cast<T>(0)) {
					gradScalar += m_WeightDecay * param[j];
				}

				// Update biased moments
				m[j] = m_Beta1 * m[j] + (static_cast<T>(1) - m_Beta1) * gradScalar;
				v[j] = m_Beta2 * v[j] + (static_cast<T>(1) - m_Beta2) * (gradScalar * gradScalar);

				// Bias correction
				T m_hat = m[j] / bias1;
				T v_hat = v[j] / bias2;

				param[j] -= m_LearningRate * (m_hat / (std::sqrt(v_hat) + m_Epsilon));
			}
		}
	}

	template <typename T>
	AdamW<T>::AdamW(std::vector<Parameter<T>>& params, T learningRate, T beta1, T beta2, T epsilon, T weightDecay)
		: Optimizer<T>(params), m_LearningRate(learningRate), m_Beta1(beta1), m_Beta2(beta2), m_BetaPow1(beta1), m_BetaPow2(beta2),
		  m_Epsilon(epsilon), m_WeightDecay(weightDecay), m_Timestep(0) {
		for (Parameter<T>& p : this->m_Params) {
			TensorCore::Tensor<T>& param = p.Data();

			TensorCore::Tensor<T> m{ param.GetShape(), param.GetAllocator() };
			TensorCore::Tensor<T> v{ param.GetShape(), param.GetAllocator() };

			m.Fill(static_cast<T>(0));
			v.Fill(static_cast<T>(0));

			m_FirstMoment.push_back(m);
			m_SecondMoment.push_back(v);
		}
	}

	template <typename T>
	void AdamW<T>::Step() {
		m_Timestep++;

		m_BetaPow1 *= m_Beta1;
		m_BetaPow2 *= m_Beta2;
		T bias1 = (static_cast<T>(1) - m_BetaPow1);
		T bias2 = (static_cast<T>(1) - m_BetaPow2);

		size_t sizeParams = this->m_Params.size();

		for (size_t i = 0; i < sizeParams; ++i) {
			TensorCore::Tensor<T>& param = this->m_Params[i].Data();

			if (!param.RequiresGrad() || !param.HasGrad()) {
				continue;
			}

			TensorCore::Tensor<T> grad = param.Grad();
			TensorCore::Tensor<T>& m = m_FirstMoment[i];
			TensorCore::Tensor<T>& v = m_SecondMoment[i];

			size_t size = param.NumElements();

			for (size_t j = 0; j < size; ++j) {
				T gradScalar = grad[j];

				// Weight decay
				if (m_WeightDecay != static_cast<T>(0)) {
					param[j] -= m_LearningRate * m_WeightDecay * param[j];
				}

				// Update biased moments
				m[j] = m_Beta1 * m[j] + (static_cast<T>(1) - m_Beta1) * gradScalar;
				v[j] = m_Beta2 * v[j] + (static_cast<T>(1) - m_Beta2) * (gradScalar * gradScalar);

				// Bias correction
				T m_hat = m[j] / bias1;
				T v_hat = v[j] / bias2;

				param[j] -= m_LearningRate * (m_hat / (std::sqrt(v_hat) + m_Epsilon));
			}
		}
	}
}