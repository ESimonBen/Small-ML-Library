// sigmoidGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class SigmoidGradFn : public GradFn<T> {
	public:
		SigmoidGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
			: GradFn<T>(a), outputTensor(b)
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
				T sigX = (*outputTensor)[i];

				gradInput[i] = gradOutput[i] * sigX * (static_cast<T>(1) - sigX);
			}

			input->Backward(gradInput);
		}

	private:
		TensorCore::Tensor<T>* outputTensor;
	};
}