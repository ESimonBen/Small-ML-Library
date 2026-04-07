// sumGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/operations/scalar/scalar.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class SumGradFn : public GradFn<T> {
	public:
		SumGradFn(TensorCore::Tensor<T>* a)
			: GradFn<T>(a), inputShape(a->GetShape())
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
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

	private:
		Utils::Shape inputShape;
	};
}