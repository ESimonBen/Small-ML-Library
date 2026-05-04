// tanhLayer.inl
#include <mlCore/operations/activations/activation.h>

namespace MLCore::NN {
	template <typename T>
	TensorCore::Tensor<T> TanhLayer<T>::Forward(const TensorCore::Tensor<T>& input) const {
		Memory::ArenaAllocator& allocator = input.GetAllocator();

		return Operations::Tanh(input, allocator);
	}
}