 /// activationLayers.h
#include <mlCore/operations/activations/activation.h>

namespace MLCore::NN {
	template <typename T>
	TensorCore::Tensor<T> ReLULayer<T>::Forward(const TensorCore::Tensor<T>& input) const {
		return Operations::ReLU(input);
	}
	
	template <typename T>
	LeakyReLULayer<T>::LeakyReLULayer(T alpha)
		: m_Alpha(alpha) {
		assert(m_Alpha >= 0);
	}
	
	template <typename T>
	TensorCore::Tensor<T> LeakyReLULayer<T>::Forward(const TensorCore::Tensor<T>& input) const {
		return Operations::LeakyReLU(input, m_Alpha);
	}
	
	template <typename T>
	T LeakyReLULayer<T>::Alpha() const {
		return m_Alpha;
	}
	
	template <typename T>
	TensorCore::Tensor<T> TanhLayer<T>::Forward(const TensorCore::Tensor<T>& input) const {
		return Operations::Tanh(input);
	}
	
	template <typename T>
	TensorCore::Tensor<T> SigmoidLayer<T>::Forward(const TensorCore::Tensor<T>& input) const {
		return Operations::Sigmoid(input);
	}
}