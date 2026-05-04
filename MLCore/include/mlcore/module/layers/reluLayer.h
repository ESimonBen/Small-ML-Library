// reluLayer.h
#pragma once
#include <mlCore/module/module.h>

namespace MLCore::NN {
	template <typename T>
	class ReLULayer : public Module<T> {
	public:
		ReLULayer() = default;

		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const override;

	};
}

#include "reluLayer.inl"