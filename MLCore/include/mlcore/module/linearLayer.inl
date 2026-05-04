// linearLayer.inl
#include <mlCore/operations/linearAlgebra/linalg.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::NN {
	template <typename T>
	LinearLayer<T>::LinearLayer(size_t in, size_t out, Memory::ArenaAllocator& allocator, Init::InitType weightInit, Init::InitType biasInit)
		: m_Weight(TensorCore::Tensor<T>{{in, out}, allocator}), m_Bias(TensorCore::Tensor<T>{{1, out}, allocator}){
		m_Weight.Data().SetRequiresGrad(true);
		m_Bias.Data().SetRequiresGrad(true);

		Init::Init(m_Weight.Data(), in, out, weightInit);
		Init::Init(m_Bias.Data(), 1, out, biasInit);
	}

	template <typename T>
	TensorCore::Tensor<T> LinearLayer<T>::Forward(const TensorCore::Tensor<T>& input) const {
		Memory::ArenaAllocator& allocator = input.GetAllocator();

		TensorCore::Tensor<T> mul = Operations::MatMultiply(input, m_Weight.Data(), allocator); // Matrix multiply weight with input
		TensorCore::Tensor<T> result = Operations::Add(mul, m_Bias.Data(), allocator); // Add the bias

		return result;
	}

	template <typename T>
	void LinearLayer<T>::CollectParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out) {
		out.push_back(std::ref(m_Weight));
		out.push_back(std::ref(m_Bias));
	}
}