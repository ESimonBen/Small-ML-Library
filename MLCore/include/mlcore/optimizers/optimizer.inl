// optimizer.inl
#include <cmath>
#include "optimizer.h"

namespace MLCore::Optimizers {
	template <typename T>
	Optimizer<T>::Optimizer(std::vector<NN::Parameter<T>>& params, T learningRate, T weightDecay) {
		m_ParamGroups.emplace_back(params, learningRate, weightDecay);
	}

	template <typename T>
	Optimizer<T>::Optimizer(std::vector<ParameterGroup<T>> groups)
		: m_ParamGroups(std::move(groups))
	{}

	template <typename T>
	void Optimizer<T>::ZeroGrad() {
		for (ParameterGroup<T>& paramGroup : m_ParamGroups) {
			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
				TensorCore::Tensor<T> param = p.Data();

				if (param.RequiresGrad()) {
					param.ZeroGrad();
				}
			}
		}
	}

	template<typename T>
	std::vector<ParameterGroup<T>>& MLCore::Optimizers::Optimizer<T>::ParamGroups() {
		return m_ParamGroups;
	}

	template <typename T>
	void Optimizer<T>::SetClipGradNorm(T maxNorm) {
		m_UseClip = true;
		m_MaxNorm = maxNorm;
	}

	template <typename T>
	void Optimizer<T>::ClipGradients() {
		if (!m_UseClip) {
			return;
		}

		T totalNorm = static_cast<T>(0);

		for (ParameterGroup<T>& paramGroup : m_ParamGroups) {
			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();
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
		}

		totalNorm = std::sqrt(totalNorm);

		if (totalNorm <= m_MaxNorm) {
			return;
		}

		T scale = m_MaxNorm / (totalNorm + static_cast<T>(1e-6));

		for (ParameterGroup<T>& paramGroup : m_ParamGroups) {
			for (auto& ref : paramGroup.params) {
				NN::Parameter<T>& p = ref.get();

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
}