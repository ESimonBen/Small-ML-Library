// transposeMulGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/operations/linearAlgebra/linalg.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class TransposeGradFn : public GradFn<T> {
	public:
		TransposeGradFn(TensorCore::Tensor<T>* a)
			: GradFn<T>(a)
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			auto* input = this->inputs[0];

			if (gradOutput.Dims()[0] != input->Dims()[1] || gradOutput.Dims()[1] != input->Dims()[0]) {
				throw std::runtime_error("ERROR: TransposeGradFn: gradOutput shape mismatch");
			}

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			if (!input->RequiresGrad()) {
				return;
			}

			auto gradInput = Operations::Transpose(gradOutput, allocator);

			input->Backward(gradInput);
		}
	};
}