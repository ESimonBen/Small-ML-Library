// module.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/parameters/parameter.h>

namespace MLCore::NN {
	template <typename T>
	class Module {
	public:
		virtual ~Module() = default;

		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) = 0;

		void Add(std::shared_ptr<Module<T>> mod);

		virtual std::vector<std::reference_wrapper<NN::Parameter<T>>> GetParameters();

	protected:
		virtual void CollectParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out);

		void CollectSubmoduleParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out) const;

	protected:
		std::vector<std::shared_ptr<Module<T>>> m_Submodules;
	};
}

#include "module.inl"