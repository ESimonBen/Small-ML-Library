 /// broadcastGradFn.inl
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
		/*TensorCore::Tensor<T> inp = input.Detach();*/

		TensorCore::Tensor<T> gradInput = Operations::Unsqueeze(gradientOut, m_Axis, allocator);

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
		/*TensorCore::Tensor<T> inp = input.Detach();*/

		TensorCore::Tensor<T> gradInput = Operations::Squeeze(gradientOut, m_Axis, allocator);

		input.Backward(gradInput);
	}
	
	template <typename T>
	ReduceToShapeGradFn<T>::ReduceToShapeGradFn(std::shared_ptr<typename GradFn<T>::Impl> a)
		: GradFn<T>(a), m_OriginalShape(a->shape)
	{}
	
	template <typename T>
	void ReduceToShapeGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> gradInput = Operations::ExpandToShape(gradientOut, m_OriginalShape, allocator);

		input.Backward(gradInput);
	}

	template <typename T>
	ExpandToShapeGradFn<T>::ExpandToShapeGradFn(std::shared_ptr<typename GradFn<T>::Impl> a)
		: GradFn<T>(a), m_OriginalShape(a->shape)
	{}

	template <typename T>
	void ExpandToShapeGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> gradInput = Operations::ReduceSumToShape(gradientOut, m_OriginalShape, allocator);

		input.Backward(gradInput);
	}
}