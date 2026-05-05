// sigmoidLayer.h
#pragma once
#include <mlCore/module/module.h>

namespace MLCore::NN {

	// Do NOT use this with BCEWithLogits, as that already uses the Sigmoid activation function internally
	template <typename T>
	class SigmoidLayer : public Module<T> {
	public:
		SigmoidLayer() = default;

		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const override;

	};
}

#include "sigmoidLayer.inl"