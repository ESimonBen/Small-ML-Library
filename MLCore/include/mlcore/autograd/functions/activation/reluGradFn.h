// reluGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class ReLUGradFn : public GradFn<T> {
	public:
		ReLUGradFn(TensorCore::Tensor<T>* a)
			: GradFn<T>(a)
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			auto* input = this->inputs[0];

			if (!input->RequiresGrad()) {
				return;
			}

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			auto& gradOutputShape = gradOutput.GetShape();

			TensorCore::Tensor<T> gradInput{ input->GetShape(), allocator};


			for (size_t i = 0; i < gradInput.NumElements(); ++i) {
				gradInput[i] = ((*input)[i] > static_cast<T>(0)) ? gradOutput[i] : static_cast<T>(0);
			}

			input->Backward(gradInput);
		}
	};	
}