// powerGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class PowerGradFn : public GradFn<T> {
	public:
		PowerGradFn(TensorCore::Tensor<T>* a, T exponent)
			: GradFn<T>(a), exponent(exponent)
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			auto* input = this->inputs[0];

			if (!input->RequiresGrad()) {
				return;
			}

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			TensorCore::Tensor<T> gradInput{ input->GetShape(), allocator};

			size_t size = gradInput.NumElements();

			T expMinus1 = std::pow(base, exponent - static_cast<T>(1));

			for (size_t i = 0; i < size; ++i) {
				T base = (*input)[i];

				gradInput[i] = gradOutput[i] * exponent * expMinus1;
			}
			
			input->Backward(gradInput);
		}

	private:
		T exponent;
	};
}