// sequential.h
#pragma once
#include <mlCore/module/module.h>

namespace MLCore::NN {
	template <typename T>
	class Sequential : public Module<T> {
	public:
		void Add(std::shared_ptr<Module<T>> mod);

		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) override;
	};
}

#include "sequential.inl"