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
				auto gradA = Operations::Divide(ReduceSumToShape(gradOutput, a->GetShape()), (*b), allocator);

				a->AccumulateGrad(gradA);
				
				if (a->HasGrad()) {
					a->Grad()->Backward(gradA);
				}
			}

			if (b->RequiresGrad()) {
				auto negGradOutput = Operations::Negate(ReduceSumToShape(gradOutput, b->GetShape()), allocator);
				auto bSquared = Operations::Square(*b, allocator);

				auto gradB = Operations::Multiply(negGradOutput, Operations::Divide((*a), bSquared, allocator), allocator);

				b->AccumulateGrad(gradB);
				
				if (b->HasGrad()) {
					b->Grad()->Backward(gradB);
				}
			}
		}

		// Maybe store the saved forward values of the tensors (needed for multiply and divide)
	};
}