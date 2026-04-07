// mulGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/autograd/gradientUtils.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class MulGradFn : public GradFn<T> {
	public:
		MulGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
			: GradFn<T>({ a, b })
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			auto* a = this->inputs[0];
			auto* b = this->inputs[1];

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			if (a->RequiresGrad()) {
				auto gradA = ReduceSumToShape(Operations::Multiply(gradOutput, *b, allocator), a->GetShape());
				a->Backward(gradA);

			}

			if (b->RequiresGrad()) {
				auto gradB = ReduceSumToShape(Operations::Multiply(gradOutput, *a, allocator), b->GetShape());
				b->Backward(gradB);

			}
		}

		// Maybe store the saved forward values of the tensors (needed for multiply and divide)
	};
}