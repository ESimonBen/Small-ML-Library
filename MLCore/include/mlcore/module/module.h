// module.h
#pragma once
#include <mlCore/parameters/parameter.h>

namespace MLCore::NN {
	template <typename T>
	class Module {
	public:
		virtual ~Module() = default;

		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const = 0;

		void Add(std::unique_ptr<Module<T>> mod);

		virtual std::vector<std::reference_wrapper<NN::Parameter<T>>> GetParameters();

		TensorCore::Tensor<T> operator()(const TensorCore::Tensor<T>& input);

	protected:
		virtual void CollectParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out);

		void CollectSubmoduleParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out);

	protected:
		std::vector<std::unique_ptr<Module<T>>> m_Submodules;
	};
}

#include "module.inl"