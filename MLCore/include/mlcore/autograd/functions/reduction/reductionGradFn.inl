// reductionGradFn.inl
#include <mlCore/operations/scalar/scalar.h>

namespace MLCore::AutoGrad {
	template <typename T>
	SumGradFn<T>::SumGradFn(TensorCore::Tensor<T>* a)
		: GradFn<T>(a), inputShape(a->GetShape())
	{}

	template <typename T>
	void SumGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: SumGradFn: Expected a 1D tensor");
		}

		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		TensorCore::Tensor<T> gradInput{ inputShape, allocator };

		T gradOutputScalar = gradOutput[0];
		size_t size = gradInput.NumElements();

		for (size_t i = 0; i < size; ++i) {
			gradInput[i] = gradOutputScalar;
		}

		input->Backward(gradInput);
	}

	// Might remove this later
	template <typename T>
	MeanGradFn<T>::MeanGradFn(TensorCore::Tensor<T>* a)
		: GradFn<T>(a), inputShape(a->GetShape()), numElements(a->NumElements())
	{}
	
	// Might remove this later as well
	template <typename T>
	void MeanGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: MeanGradFn: Expected a 1D tensor");
		}

		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		TensorCore::Tensor<T> gradInput{ inputShape, allocator };

		T gradScalar = gradOutput[0] / static_cast<T>(numElements);

		for (size_t i = 0; i < numElements; ++i) {
			gradInput[i] = gradScalar;
		}

		input->Backward(gradInput);
	}

	template <typename T>
	MaxGradFn<T>::MaxGradFn(TensorCore::Tensor<T>* a, T maxValue)
		: GradFn<T>(a), inputShape(a->GetShape()), maxValue(maxValue)
	{}

	template <typename T>
	void MaxGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: MaxGradFn: Expected a 1D tensor");
		}

		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		TensorCore::Tensor<T> gradInput{ inputShape, allocator };

		T gradScalar = gradOutput[0];

		size_t size = gradInput.NumElements();

		// May change implementation to split the gradient over all the max elements
		// in the tensor rather than setting all of the max elements to the max gradient (for stability's sake)
		for (size_t i = 0; i < size; ++i) {
			gradInput[i] = ((*input)[i] == maxValue) gradScalar : 0
		}

		input->Backward(gradInput);
	}

	template <typename T>
	MinGradFn<T>::MinGradFn(TensorCore::Tensor<T>* a, T minValue)
		: GradFn<T>(a), inputShape(a->GetShape()), minValue(minValue)
	{}

	template <typename T>
	void MinGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: MinGradFn: Expected a 1D tensor");
		}

		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		TensorCore::Tensor<T> gradInput{ inputShape, allocator };

		T gradScalar = gradOutput[0];

		size_t size = gradInput.NumElements();

		// May change implementation to split the gradient over all the max elements in the tensor 
		// rather than setting all of the max elements to the max gradient (for stability's sake)
		for (size_t i = 0; i < size; ++i) {
			gradInput[i] = ((*input)[i] == minValue) gradScalar : 0
		}

		input->Backward(gradInput);
	}

	template <typename T>
	AxisSumGradFn<T>::AxisSumGradFn(TensorCore::Tensor<T>* a, size_t axis)
		: GradFn<T>(a), axis(axis), inputShape(a->GetShape()) {
		assert(axis < inputShape.Rank());
	}

	template <typename T>
	void AxisSumGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		auto& gradOutputShape = gradOutput.GetShape();

		TensorCore::Tensor<T> gradInput{ inputShape, allocator };

		size_t size = gradInput.NumElements();

		for (size_t i = 0; i < size; ++i) {
			auto reduced = inputShape.UnflattenIndex(i);
			reduced.erase(reduced.begin() + axis);

			if (reduced.empty()) {
				reduced.push_back(0);
			}

			size_t gradOutputIndex = gradOutputShape.FlattenIndex(reduced);

			gradInput[i] = gradOutput[gradOutputIndex];
		}

		input->Backward(gradInput);
	}
}