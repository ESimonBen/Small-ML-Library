// linearLayer.inl
#include <mlCore/operations/linearAlgebra/linalg.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::NN {
	template <typename T>
	LinearLayer<T>::LinearLayer(size_t in, size_t out, Memory::ArenaAllocator& allocator) {
		m_Weight = Parameter<T>{ TensorCore::Tensor<T>{{in, out}, allocator} };
		m_Bias = Parameter<T>{ TensorCore::Tensor<T>{{1, out}, allocator} };

		// Initialize to prevent garbage values (soon to be replaced with Xavier and/or He initialization)
		m_Weight.Data().Fill(static_cast<T>(0));
		m_Bias.Data().Fill(static_cast<T>(0));

		m_Weight.Data().SetRequiresGrad(true);
		m_Bias.Data().SetRequiresGrad(true);
	}

	template <typename T>
	TensorCore::Tensor<T> LinearLayer<T>::Forward(const TensorCore::Tensor<T>& input) {
		auto& allocator = input.GetAllocator();

		TensorCore::Tensor<T> mul = Operations::MatMultiply(input, m_Weight.Data(), allocator); // Matrix multiply weight with input
		TensorCore::Tensor<T> result = Operations::Add(mul, m_Bias.Data(), allocator); // Add the bias

		return result;
	}

	template <typename T>
	void LinearLayer<T>::CollectParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out) const {
		out.push_back(m_Weight);
		out.push_back(m_Bias);
	}
}