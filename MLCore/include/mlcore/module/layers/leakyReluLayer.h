// leakyReluLayer.h
#pragma once
#include <mlCore/module/module.h>

namespace MLCore::NN {
	template <typename T>
	class LeakyReLULayer : public Module<T> {
	public:
		LeakyReLULayer(T alpha = static_cast<T>(.01));

		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const override;

		T Alpha() const;

	private:
		const T m_Alpha;
	};
}

#include "leakyReluLayer.inl"