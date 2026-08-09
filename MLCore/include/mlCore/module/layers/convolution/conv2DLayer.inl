/// conv2DLayer.inl
#include <mlCore/operations/convolution/convolution.h>

namespace MLCore::NN {
	template <typename T>
	inline Conv2DLayer<T>::Conv2DLayer(size_t inChannels, size_t outChannels, size_t kernelHeight, size_t kernelWidth,
		size_t strideH, size_t strideW, size_t paddingH, size_t paddingW, size_t dilationH, size_t dilationW,
		InitType kernelInit, InitType biasInit)
		: m_Kernel(TensorCore::Tensor<T>{{inChannels, outChannels, kernelHeight, kernelWidth}}), m_Bias(TensorCore::Tensor<T>{{outChannels}}),
		  m_StrideH(strideH), m_StrideW(strideW), m_PaddingH(paddingH), m_PaddingW(paddingW), m_DilationH(dilationH), m_DilationW(dilationW) {
		m_Kernel.Data().SetRequiresGrad(true);
		m_Bias.Data().SetRequiresGrad(true);

		const size_t fanIn = inChannels * kernelHeight * kernelWidth;
		const size_t fanOut = outChannels * kernelHeight * kernelWidth;

		Init(m_Kernel.Data(), fanIn, fanOut, kernelInit);
		Init(m_Bias.Data(), 1, outChannels, biasInit);
	}

	template <typename T>
	inline TensorCore::Tensor<T> Conv2DLayer<T>::Forward(const TensorCore::Tensor<T>& input) const {
		return Operations::Conv2D(input, m_Kernel.Data(), &(m_Bias.Data()), m_StrideH, m_StrideW, m_PaddingH, m_PaddingW, m_DilationH, m_DilationW);
	}

	template <typename T>
	inline void Conv2DLayer<T>::CollectParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out) {
		out.push_back(std::ref(m_Kernel));
		out.push_back(std::ref(m_Bias));
	}

	template <typename T>
	inline void Conv2DLayer<T>::CollectParameters(std::vector<std::reference_wrapper<const NN::Parameter<T>>>& out) const {
		out.push_back(std::cref(m_Kernel));
		out.push_back(std::cref(m_Bias));
	}

	template <typename T>
	inline void Conv2DLayer<T>::CollectNamedParameters(const std::string& name, std::vector<NamedParameter<T>>& out) {
		auto MakeName = [&](const std::string& suffix) {
			return (name.empty()) ? suffix : name + "." + suffix;
		};

		out.emplace_back(MakeName("kernel"), std::ref(m_Kernel));
		out.emplace_back(MakeName("bias"), std::ref(m_Bias));
	}

	template <typename T>
	inline void Conv2DLayer<T>::CollectNamedParameters(const std::string& name, std::vector<ConstNamedParameter<T>>& out) const {
		auto MakeName = [&](const std::string& suffix) {
			return (name.empty()) ? suffix : name + "." + suffix;
		};

		out.emplace_back(MakeName("kernel"), std::ref(m_Kernel));
		out.emplace_back(MakeName("bias"), std::ref(m_Bias));
	}
}