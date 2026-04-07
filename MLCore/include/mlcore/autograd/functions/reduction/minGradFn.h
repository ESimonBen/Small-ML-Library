// minGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/operations/scalar/scalar.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class MinGradFn : public GradFn<T> {
	public:
		MinGradFn(TensorCore::Tensor<T>* a, T minValue)
			: GradFn<T>(a), inputShape(a->GetShape()), minValue(minValue)
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
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

			// May change implementation to split the gradient over all the max elements
			// in the tensor rather than setting all of the max elements to the max gradient (for stability's sake)
			for (size_t i = 0; i < size; ++i) {
				gradInput[i] = ((*input)[i] == minValue) gradScalar : 0
			}

			input->Backward(gradInput);
		}

	private:
		Utils::Shape inputShape;
		T minValue;
	};
}