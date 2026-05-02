// sequential.h
#pragma once
#include <mlCore/module/module.h>

namespace MLCore::NN {
	template <typename T>
	class Sequential : public Module<T> {
	public:
		void Add(std::shared_ptr<Module<T>> mod);

		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) override;

	/*protected:
		virtual void Parameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out) const override;*/

	/*private:
		std::vector<std::shared_ptr<Module<T>>> m_Layers;*/
	};
}

#include "sequential.inl"