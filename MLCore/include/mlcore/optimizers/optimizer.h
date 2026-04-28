// optimizer.h
#pragma once
#include <vector>
#include <functional>
#include <mlCore/tensor/tensor.h>

namespace MLCore::Optimizers {
	template <typename T>
	struct Parameter {
		TensorCore::Tensor<T> data;

		Parameter() = default;
		Parameter(const Parameter&) = delete;
		Parameter& operator=(const Parameter&) = delete;
		/*Parameter(Parameter&&) = delete;
		Parameter& operator=(Parameter&&) = delete;*/

		explicit Parameter(const TensorCore::Tensor<T>& tensor)
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
	struct ParameterGroup {
		std::vector<Parameter<T>*> params; // I feel like this will cause problems soon

		T learningRate;
		T weightDecay;

		ParameterGroup(std::vector<Parameter<T>>& paramsVec, T learningRate, T weightDecay = static_cast<T>(0))
			: learningRate(learningRate), weightDecay(weightDecay) {
			params.reserve(paramsVec.size());

			for (Parameter<T>& p : paramsVec) {
				params.push_back(&p);
			}
		}
	};

	template <typename T>
	class Optimizer {
	public:
		Optimizer(std::vector<Parameter<T>>& params, T learningRate, T weightDecay = static_cast<T>(0));
		Optimizer(std::vector<ParameterGroup<T>> groups);

		virtual ~Optimizer() = default;

		// Update rule (changes with different optimizers)
		virtual void Step() = 0;
		virtual void ZeroGrad();

		std::vector<ParameterGroup<T>>& ParamGroups();
		void SetClipGradNorm(T maxNorm);

	protected:
		// Functions
		void ClipGradients();
		
		// Members
		std::vector<ParameterGroup<T>> m_ParamGroups;

	private:
		bool m_UseClip = false;
		T m_MaxNorm = static_cast<T>(0);
	};
}

#include "optimizer.inl"