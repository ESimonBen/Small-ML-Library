// broadcastGradFn.inl
#include <mlCore/operations/broadcast/broadcast.h>

namespace MLCore::AutoGrad {
	template <typename T>
	SqueezeGradFn<T>::SqueezeGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis)
		: GradFn<T>(a), m_Axis(axis)
	{}

	template <typename T>
	void SqueezeGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> inp = input.Detach();

		TensorCore::Tensor<T> gradInput = Operations::Unsqueeze(inp, m_Axis, allocator);

		input.Backward(gradInput);
	}

	template <typename T>
	UnsqueezeGradFn<T>::UnsqueezeGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis)
		: GradFn<T>(a), m_Axis(axis)
	{}

	template <typename T>
	void UnsqueezeGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> inp = input.Detach();

		TensorCore::Tensor<T> gradInput = Operations::Squeeze(inp, m_Axis, allocator);

		input.Backward(gradInput);
	}
}