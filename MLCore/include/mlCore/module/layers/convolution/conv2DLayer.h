/// conv2DLayer.h
#include <mlCore/module/module.h>
#include <mlCore/parameters/initialization.h>

namespace MLCore::NN {
	template <typename T>
	class Conv2DLayer : public Module<T> {
	public:
		Conv2DLayer(size_t inChannels, size_t outChannels, size_t kernelHeight, size_t kernelWidth, 
					size_t strideH = 1, size_t strideW = 1, size_t paddingH = 0, size_t paddingW = 0, size_t dilationH = 1, size_t dilationW = 1, 
					InitType kernelInit = InitType::XavierUniform, InitType biasInit = InitType::Zero);

		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const override;

	protected:
		virtual void CollectParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out) override;

		virtual void CollectParameters(std::vector<std::reference_wrapper<const NN::Parameter<T>>>& out) const override;

		virtual void CollectNamedParameters(const std::string& name, std::vector<NamedParameter<T>>& out) override;

		virtual void CollectNamedParameters(const std::string& name, std::vector<ConstNamedParameter<T>>& out) const override;

	private:
		Parameter<T> m_Kernel; /// Member variable that holds a kernel parameter as a Parameter<T> instance.
		Parameter<T> m_Bias; /// Member variable that holds a bias parameter as a Parameter<T> instance.

		size_t m_StrideH, m_StrideW; /// Stride along height and width dimensions for the convolution operation.

		size_t m_PaddingH, m_PaddingW; /// Padding along height and width dimensions for the convolution operation.

		size_t m_DilationH, m_DilationW; /// Dilation along height and width dimensions for the convolution operation.
	};
}

#include "conv2DLayer.inl"