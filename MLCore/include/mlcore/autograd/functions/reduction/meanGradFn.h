// meanGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/operations/scalar/scalar.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class MeanGradFn : public GradFn<T> {
	public:
		MeanGradFn(TensorCore::Tensor<T>* a)
			: GradFn<T>(a), inputShape(a->GetShape()), numElements(a->NumElements())
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			if (gradOutput.NumElements() != 1) {
				throw std::runtime_error("ERROR: MeanGradFn: Expected a 1D tensor");
			}

			auto* input = this->inputs[0];

			if (!input->RequiresGrad()) {
				return;
			}

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			TensorCore::Tensor<T> gradInput{ inputShape, allocator };

			T gradScalar = gradOutput[0] / static_cast<T>(numElements);

			for (size_t i = 0; i < numElements; ++i) {
				gradInput[i] = gradScalar;
			}

			input->Backward(gradInput);
		}

	private:
		Utils::Shape inputShape;
		size_t numElements;
	};
}