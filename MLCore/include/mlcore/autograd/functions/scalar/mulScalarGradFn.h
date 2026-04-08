// mulScalarGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class MulScalarGradFn : public GradFn<T> {
	public:
		MulScalarGradFn(TensorCore::Tensor<T>* a, T scalar)
			: GradFn<T>(a), scalar(scalar)
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			auto* input = this->inputs[0];

			if (!input->RequiresGrad()) {
				return;
			}

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			TensorCore::Tensor<T> gradInput{ input->GetShape(), allocator };
			size_t size = gradInput.NumElements();

			for (size_t i = 0; i < size; ++i) {
				gradInput[i] = gradOutput[i] * scalar;
			}

			input->Backward(gradInput);
		}

	private:
		T scalar;
	};
}