// divGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/autograd/gradientUtils.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class DivGradFn : public GradFn<T> {
	public:
		DivGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
			: GradFn<T>({ a, b })
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			auto* a = this->inputs[0];
			auto* b = this->inputs[1];

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			if (a->RequiresGrad()) {
				auto gradA = ReduceSumToShape(Operations::Divide(gradOutput, *b, allocator), a->GetShape());
				a->Backward(gradA);

			}

			if (b->RequiresGrad()) {
				auto negGradOutput = Operations::Negate(gradOutput, allocator);
				auto bSquared = Operations::Square(*b, allocator);

				auto gradB = ReduceSumToShape(Operations::Multiply(negGradOutput, Operations::Divide((*a), bSquared, allocator), allocator), b->GetShape());
				b->Backward(gradB);

			}
		}

		// Maybe store the saved forward values of the tensors (needed for multiply and divide)
	};
}