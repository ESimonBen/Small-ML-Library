 /// linearLayer.inl
#include <mlCore/operations/broadcast/broadcast.h>
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
		if (input.Rank() != 2) {
			const std::vector<size_t>& dims = input.Dims();
			size_t rank = input.Rank();

			std::vector<size_t> finalDims;
			finalDims.reserve(rank); /// Reserve space for leading dimensions and output features (from weight)
			size_t combinedSize = 1; /// Multiplier for combined size

			for (size_t i = 0; i < rank - 1; ++i) {
				finalDims.push_back(dims[i]); /// Place leading dimensions
				combinedSize *= dims[i]; /// Combine leading dimensions into 1 size
			}

			finalDims.push_back(m_Weight.Data().Dims()[1]); /// Place output features from weight

			TensorCore::Tensor<T> input2D = Operations::Reshape(input, Utils::Shape{ combinedSize, dims[rank - 1] });
			TensorCore::Tensor<T> mul = Operations::MatMultiply(input2D, m_Weight.Data()); /// Matrix multiply weight with input
			TensorCore::Tensor<T> result2D = Operations::Add(mul, m_Bias.Data()); /// Add the bias
			TensorCore::Tensor<T> result = Operations::Reshape(result2D, Utils::Shape{ finalDims });
			
			return result;
		}
		else {
			TensorCore::Tensor<T> mul = Operations::MatMultiply(input, m_Weight.Data()); /// Matrix multiply weight with input
			TensorCore::Tensor<T> result = Operations::Add(mul, m_Bias.Data()); /// Add the bias

			return result;
		}
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