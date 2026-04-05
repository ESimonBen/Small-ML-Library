// divGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
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

			if (a->RequiresGrad()) {
				TensorCore::Tensor<T> divOutput = Operations::Divide(gradOutput, (*b), gradOutput.GetAllocator());

				a->AccumulateGrad(divOutput);
				a->Backward(divOutput);
			}

			if (b->RequiresGrad()) {
				TensorCore::Tensor<T> divOutput = Operations::Multiply(Operations::Negate(gradOutput, gradOutput.GetAllocator()), Operations::Divide((*a), Operations::Square((*b), gradOutput.GetAllocator()), gradOutput.GetAllocator()), gradOutput.GetAllocator());

				b->AccumulateGrad(divOutput);
				b->Backward(divOutput);
			}
		}
	};
}