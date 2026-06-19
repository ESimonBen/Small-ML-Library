 /// activationLayers.h
#include <mlCore/operations/activations/activation.h>

namespace MLCore::NN {
	template <typename T>
	TensorCore::Tensor<T> ReLULayer<T>::Forward(const TensorCore::Tensor<T>& input) const {
		Memory::ArenaAllocator& allocator = input.GetAllocator();

		return Operations::ReLU(input, allocator);
	}
	
	template <typename T>
	LeakyReLULayer<T>::LeakyReLULayer(T alpha)
		: m_Alpha(alpha) {
		assert(m_Alpha >= 0);
	}
	
	template <typename T>
	TensorCore::Tensor<T> LeakyReLULayer<T>::Forward(const TensorCore::Tensor<T>& input) const {
		Memory::ArenaAllocator& allocator = input.GetAllocator();

		return Operations::LeakyReLU(input, m_Alpha, allocator);
	}
	
	template <typename T>
	T LeakyReLULayer<T>::Alpha() const {
		return m_Alpha;
	}
	
	template <typename T>
	TensorCore::Tensor<T> TanhLayer<T>::Forward(const TensorCore::Tensor<T>& input) const {
		Memory::ArenaAllocator& allocator = input.GetAllocator();

		return Operations::Tanh(input, allocator);
	}
	
	template <typename T>
	TensorCore::Tensor<T> SigmoidLayer<T>::Forward(const TensorCore::Tensor<T>& input) const {
		Memory::ArenaAllocator& allocator = input.GetAllocator();

		return Operations::Sigmoid(input, allocator);
	}
}