// linearLayer.h
#pragma once
#include <mlCore/module/module.h>
#include <mlCore/parameters/initialization.h>

namespace MLCore::NN {
	template <typename T>
	class LinearLayer : public Module<T> {
	public:
		LinearLayer(size_t in, size_t out, Memory::ArenaAllocator& allocator, Init::InitType weightInit = Init::InitType::XavierUniform, Init::InitType biasInit = Init::InitType::Zero);

		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const override;

	protected:
		virtual void CollectParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out) override;

	private:
		Parameter<T> m_Weight;
		Parameter<T> m_Bias;
	};
}

#include "linearLayer.inl"