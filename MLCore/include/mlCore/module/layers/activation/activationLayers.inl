 /// activationLayers.h
#include <mlCore/operations/activations/activation.h>

namespace MLCore::NN {
	template <typename T>
	inline TensorCore::Tensor<T> ReLULayer<T>::Forward(const TensorCore::Tensor<T>& input) {
		return Operations::ReLU(input);
	}
	
	template <typename T>
	inline LeakyReLULayer<T>::LeakyReLULayer(T alpha)
		: m_Alpha(alpha) {
		if (m_Alpha < static_cast<T>(0)) {
			throw std::invalid_argument("ERROR: LeakyReLULayer: Alpha must be non-negative");
		}
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> LeakyReLULayer<T>::Forward(const TensorCore::Tensor<T>& input) {
		return Operations::LeakyReLU(input, m_Alpha);
	}
	
	template <typename T>
	inline T LeakyReLULayer<T>::Alpha() const {
		return m_Alpha;
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> TanhLayer<T>::Forward(const TensorCore::Tensor<T>& input) {
		return Operations::Tanh(input);
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> SigmoidLayer<T>::Forward(const TensorCore::Tensor<T>& input) {
		return Operations::Sigmoid(input);
	}
}