 /// linearLayer.inl
#include <mlCore/operations/linearAlgebra/linalg.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::NN {
	template <typename T>
	inline LinearLayer<T>::LinearLayer(size_t in, size_t out, InitType weightInit, InitType biasInit)
		: m_Weight(Utils::Shape{ in, out }), m_Bias(Utils::Shape{ 1, out }) {
		Init(m_Weight.Data(), in, out, weightInit);
		Init(m_Bias.Data(), 1, out, biasInit);
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> LinearLayer<T>::Forward(const TensorCore::Tensor<T>& input) {
		TensorCore::Tensor<T> mul = Operations::MatMultiply(input, m_Weight.Data()); /// Matrix multiply weight with input
		TensorCore::Tensor<T> result = Operations::Add(mul, m_Bias.Data()); /// Add the bias

		return result;
	}
	
	template <typename T>
	inline void LinearLayer<T>::CollectNamedParameters(const std::string& name, std::vector<NamedParameter<T>>& out) {
		auto MakeName = [&](const std::string& suffix) {
			return (name.empty()) ? suffix : name + "." + suffix;
		};

		out.emplace_back(MakeName("weight"), std::ref(m_Weight));
		out.emplace_back(MakeName("bias"), std::ref(m_Bias));
	}
	
	template <typename T>
	inline void LinearLayer<T>::CollectNamedParameters(const std::string& name, std::vector<ConstNamedParameter<T>>& out) const {
		auto MakeName = [&](const std::string& suffix) {
			return (name.empty()) ? suffix : name + "." + suffix;
		};

		out.emplace_back(MakeName("weight"), std::ref(m_Weight));
		out.emplace_back(MakeName("bias"), std::ref(m_Bias));
	}
}