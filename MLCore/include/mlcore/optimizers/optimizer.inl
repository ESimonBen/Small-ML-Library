// optimizer.inl
#include <cmath>
#include "optimizer.h"

namespace MLCore::Optimizers {
	template <typename T>
	Optimizer<T>::Optimizer(std::vector<Parameter<T>>& params, T learningRate)
		: m_Params(params), m_LearningRate(learningRate)
	{}

	template <typename T>
	void Optimizer<T>::ZeroGrad() {
		for (Parameter<T>& p : m_Params) {
			TensorCore::Tensor<T> param = p.Data();

			if (param.RequiresGrad()) {
				param.ZeroGrad();
			}
		}
	}

	template<typename T>
	std::vector<TensorCore::Tensor<T>>& MLCore::Optimizers::Optimizer<T>::Params() {
		return m_Params;
	}

	template <typename T>
	void Optimizer<T>::SetClipGradNorm(T maxNorm) {
		m_UseClip = true;
		m_MaxNorm = maxNorm;
	}

	template <typename T>
	T Optimizer<T>::LearningRate() {
		return m_LearningRate;
	}

	template <typename T>
	void Optimizer<T>::SetLearningRate(T learningRate) {
		m_LearningRate = learningRate;
	}

	template <typename T>
	void Optimizer<T>::ClipGradients() {
		if (!m_UseClip) {
			return;
		}

		T totalNorm = static_cast<T>(0);

		for (Parameter<T>& p : m_Params) {
			TensorCore::Tensor<T>& param = p.Data();
			if (!param.HasGrad()) {
				continue;
			}

			TensorCore::Tensor<T> grad = param.Grad();
			size_t size = grad.NumElements();

			for (size_t i = 0; i < size; ++i) {
				totalNorm += grad[i] * grad[i];
			}
		}

		totalNorm = std::sqrt(totalNorm);

		if (totalNorm <= m_MaxNorm) {
			return;
		}

		T scale = m_MaxNorm / (totalNorm + static_cast<T>(1e-6));

		for (Parameter<T>& p : m_Params) {
			TensorCore::Tensor<T>& param = p.Data();
			if (!param.HasGrad()) {
				continue;
			}

			TensorCore::Tensor<T> grad = param.Grad();
			size_t size = grad.NumElements();

			for (size_t i = 0; i < size; ++i) {
				grad[i] *= scale;
			}
		}
	}
}