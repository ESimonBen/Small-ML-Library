// negateGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/autograd/gradientUtils.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class NegateGradFn : public GradFn<T> {
	public:
		NegateGradFn(TensorCore::Tensor<T>* a)
			: GradFn<T>(a)
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			auto* input = this->inputs[0];

			if (!input->RequiresGrad()) {
				return;
			}

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			auto gradInput = Operations::Negate(gradOutput, allocator);

			input->Backward(gradInput);
		}
	};
}