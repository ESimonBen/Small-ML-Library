// sigmoidLayer.inl
#include <mlCore/operations/activations/activation.h>

namespace MLCore::NN {
	template <typename T>
	TensorCore::Tensor<T> SigmoidLayer<T>::Forward(const TensorCore::Tensor<T>& input) const {
		Memory::ArenaAllocator& allocator = input.GetAllocator();

		return Operations::Sigmoid(input, allocator);
	}
}