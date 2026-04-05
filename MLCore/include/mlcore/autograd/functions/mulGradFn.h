// mulGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
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

			if (a->RequiresGrad()) {
				TensorCore::Tensor<T> mulOutput = Operations::Multiply(gradOutput, (*b), gradOutput.GetAllocator());

				a->AccumulateGrad(mulOutput);
				a->Backward(mulOutput);
			}

			if (b->RequiresGrad()) {
				TensorCore::Tensor<T> mulOutput = Operations::Multiply(gradOutput, (*a), gradOutput.GetAllocator());

				b->AccumulateGrad(mulOutput);
				b->Backward(mulOutput);
			}
		}
	};
}