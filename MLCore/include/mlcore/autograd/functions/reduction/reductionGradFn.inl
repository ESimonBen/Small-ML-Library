// reductionGradFn.inl
#include <stdexcept>
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/broadcast/broadcast.h>

namespace MLCore::AutoGrad {
	template <typename T>
	SumGradFn<T>::SumGradFn(std::shared_ptr<typename GradFn<T>::Impl> a)
		: GradFn<T>(a), inputShape(a->shape)
	{}

	template <typename T>
	void SumGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: SumGradFn: Expected scalar tensor");
		}

		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor gradientOut = gradOutput.Detach();

		auto gradInput = ExpandToShape(gradientOut, inputShape);

		input.Backward(gradInput);
	}

	template <typename T>
	MaxGradFn<T>::MaxGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T maxValue)
		: GradFn<T>(a), inputShape(a->shape), maxValue(maxValue)
	{}

	template <typename T>
	void MaxGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: MaxGradFn: Expected a 1D tensor");
		}

		TensorCore::Tensor<T> input{this->inputs[0]};

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		TensorCore::Tensor<T> gradInput{ inputShape, allocator };

		T gradScalar = gradOutput[0];

		size_t size = gradInput.NumElements();
		size_t count = 0;

		// May change implementation to split the gradient over all the max elements
		// in the tensor rather than setting all of the max elements to the max gradient (for stability's sake)
		for (size_t i = 0; i < size; ++i) {
			if (input[i] == maxValue) {
				count++;
			}
		}

		for (size_t i = 0; i < size; ++i) {
			gradInput[i] = (input[i] == maxValue) ? gradScalar / count : 0;
		}

		input.Backward(gradInput);
	}

	template <typename T>
	MinGradFn<T>::MinGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T minValue)
		: GradFn<T>(a), inputShape(a->shape), minValue(minValue)
	{}

	template <typename T>
	void MinGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: MinGradFn: Expected a 1D tensor");
		}

		TensorCore::Tensor<T> input{this->inputs[0]};

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		TensorCore::Tensor<T> gradInput{ inputShape, allocator };

		T gradScalar = gradOutput[0];

		size_t size = gradInput.NumElements();
		size_t count = 0;

		// May change implementation to split the gradient over all the max elements in the tensor 
		// rather than setting all of the max elements to the max gradient (for stability's sake)
		for (size_t i = 0; i < size; ++i) {
			if (input[i] == minValue) {
				count++;
			}
		}

		for (size_t i = 0; i < size; ++i) {
			gradInput[i] = (input[i] == minValue) ? gradScalar / count : 0;
		}

		input.Backward(gradInput);
	}

	template <typename T>
	AxisSumGradFn<T>::AxisSumGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis)
		: GradFn<T>(a), inputShape(a->shape) {
		assert(axis < inputShape.Rank());
	}

	template <typename T>
	void AxisSumGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input{this->inputs[0]};

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> grad = gradOutput.Detach();
		auto& allocator = grad.GetAllocator();

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		TensorCore::Tensor<T> gradInput = ExpandToShape(gradientOut, inputShape);

		input.Backward(gradInput);
	}
}