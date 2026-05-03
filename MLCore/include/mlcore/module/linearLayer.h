// linearLayer.h
#pragma once
#include <mlCore/module/module.h>

namespace MLCore::NN {
	template <typename T>
	class LinearLayer : public Module<T> {
	public:
		LinearLayer(size_t in, size_t out, Memory::ArenaAllocator& allocator);

		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) override;

	protected:
		virtual void CollectParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out) override;

	private:
		Parameter<T> m_Weight;
		Parameter<T> m_Bias;
	};
}

#include "linearLayer.inl"