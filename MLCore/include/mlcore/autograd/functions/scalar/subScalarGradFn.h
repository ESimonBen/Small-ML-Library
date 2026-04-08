// subScalarGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class SubScalarGradFn : public GradFn<T> {
	public:
		SubScalarGradFn(TensorCore::Tensor<T>* a, bool scalarOnLeft)
			: GradFn<T>(a), scalarOnLeft(scalarOnLeft)
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			auto* input = this->inputs[0];

			if (!input->RequiresGrad()) {
				return;
			}

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			TensorCore::Tensor<T> gradInput{ input->GetShape(), allocator };
			size_t size = gradInput.NumElements();

			T sign = (scalarOnLeft) ? static_cast<T>(-1) : static_cast<T>(1);

			for (size_t i = 0; i < size; ++i) {
				gradInput[i] = gradOutput[i] * sign;
			}

			input->Backward(gradInput);
		}

	private:
		bool scalarOnLeft;
	};
}