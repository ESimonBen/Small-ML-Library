// optimizer.h
#pragma once
#include <vector>
#include <mlCore/tensor/tensor.h>

namespace MLCore::Optimizers {
	template <typename T>
	struct Parameter {
		TensorCore::Tensor<T> data;

		Parameter() = default;

		Parameter(const TensorCore::Tensor<T>& tensor)
			: data(tensor)
		{}

		TensorCore::Tensor<T>& Data() {
			return data;
		}

		const TensorCore::Tensor<T>& Data() const {
			return data;
		}

		T* RawData() {
			return data.Data();
		}

		const T* RawData() const {
			return data.Data();
		}
	};

	template <typename T>
	class Optimizer {
	public:
		Optimizer(std::vector<Parameter<T>>& params);
		virtual ~Optimizer() = default;

		// Update rule (changes with different optimizers)
		virtual void Step() = 0;
		virtual void ZeroGrad();

		std::vector<TensorCore::Tensor<T>>& Params();

	protected:
		std::vector<Parameter<T>>& m_Params;
	};
}

#include "optimizer.inl"