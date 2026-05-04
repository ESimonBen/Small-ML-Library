// tanhLayer.h
#pragma once
#include <mlCore/module/module.h>

namespace MLCore::NN {
	template <typename T>
	class TanhLayer : public Module<T> {
	public:
		TanhLayer() = default;

		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const override;

	};
}

#include "tanhLayer.inl"