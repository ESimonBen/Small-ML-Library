// optimizer.inl
#include "optimizer.h"

namespace MLCore::Optimizers {
	template <typename T>
	Optimizer<T>::Optimizer(std::vector<Parameter<T>>& params)
		: m_Params(params)
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
	inline std::vector<TensorCore::Tensor<T>>& MLCore::Optimizers::Optimizer<T>::Params() {
		return m_Params;
	}
}