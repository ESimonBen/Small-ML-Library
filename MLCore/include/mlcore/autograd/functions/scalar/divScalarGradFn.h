// divScalarGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class DivScalarGradFn : public GradFn<T> {
	public:
		DivScalarGradFn(TensorCore::Tensor<T>* a, T scalar, bool scalarOnLeft)
			: GradFn<T>(a), scalar(scalar), scalarOnLeft(scalarOnLeft)
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			auto* input = this->inputs[0];

			if (!input->RequiresGrad()) {
				return;
			}

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			TensorCore::Tensor<T> gradInput{ input->GetShape(), allocator };
			size_t size = gradInput.NumElements();

			// Should either be using elementwise or scalar operations here (or both)
			if (!scalarOnLeft) {
				T inverseScalar = static_cast<T>(1) / scalar;
				for (size_t i = 0; i < size; ++i) {
					gradInput[i] = gradOutput[i] * inverseScalar;
				}
			}
			else {
				for (size_t i = 0; i < size; ++i) {
					T inputScalar = (*input)[0];
					gradInput[i] = gradOutput[i] * (-scalar / (inputScalar * inputScalar));
				}
			}

			input->Backward(gradInput);
		}

	private:
		T scalar;
		bool scalarOnLeft;
	};
}