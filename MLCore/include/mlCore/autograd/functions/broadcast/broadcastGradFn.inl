 /// broadcastGradFn.inl
#include <mlCore/operations/broadcast/broadcast.h>

namespace MLCore::AutoGrad {
	template <typename T>
	inline SqueezeGradFn<T>::SqueezeGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> a, size_t axis)
		: GradFn<T>(a), m_Axis(axis)
	{}
	
	template <typename T>
	inline void SqueezeGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		TensorCore::Tensor<T> gradInput = Operations::Unsqueeze(gradientOut, m_Axis);

		input.Backward(gradInput);
	}
	
	template <typename T>
	inline UnsqueezeGradFn<T>::UnsqueezeGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> a, size_t axis)
		: GradFn<T>(a), m_Axis(axis)
	{}
	
	template <typename T>
	inline void UnsqueezeGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		TensorCore::Tensor<T> gradInput = Operations::Squeeze(gradientOut, m_Axis);

		input.Backward(gradInput);
	}
	
	template <typename T>
	inline ReduceToShapeGradFn<T>::ReduceToShapeGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> a)
		: GradFn<T>(a)
	{}
	
	template <typename T>
	inline void ReduceToShapeGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> gradInput = Operations::ExpandToShape(gradientOut, input.GetShape());

		input.Backward(gradInput);
	}

	template <typename T>
	inline ExpandToShapeGradFn<T>::ExpandToShapeGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> a)
		: GradFn<T>(a)
	{}

	template <typename T>
	inline void ExpandToShapeGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> gradInput = Operations::ReduceSumToShape(gradientOut, input.GetShape());

		input.Backward(gradInput);
	}
	
	template <typename T>
	inline ReshapeGradFn<T>::ReshapeGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> a)
		: GradFn<T>(a)
	{}
	
	template <typename T>
	inline void ReshapeGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> gradInput = Operations::Reshape(gradientOut, input.GetShape());

		input.Backward(gradInput);
	}
}