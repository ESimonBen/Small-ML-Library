// softmaxGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class SoftmaxGradFn : public GradFn<T> {
	public:
		SoftmaxGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
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

			T sum = 0;

			for (size_t i = 0; i < size; ++i) {
				sum += gradOutput[i] * (*outputTensor)[i];
			}

			// Should probably use scalar subtraction for this
			for (size_t i = 0; i < size; ++i) {
				gradInput[i] = (*outputTensor)[i] * (gradOutput[i] - sum);
			}

			input->Backward(gradInput);
		}

	private:
		TensorCore::Tensor<T>* outputTensor;
	};
}