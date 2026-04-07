// leakyReluGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class LeakyReLUGradFn : public GradFn<T> {
	public:
		LeakyReLUGradFn(TensorCore::Tensor<T>* a, T alpha)
			: GradFn<T>(a), alpha(alpha)
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
				gradInput[i] = ((*input)[i] > static_cast<T>(0)) ? gradOutput[i] : alpha * gradOutput[i];
			}

			input->Backward(gradInput);
		}

	private:
		T alpha;
	};
}