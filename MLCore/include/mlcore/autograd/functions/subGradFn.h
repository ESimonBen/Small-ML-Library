// subGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class SubGradFn : public GradFn<T> {
	public:
		SubGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
			: GradFn<T>({ a, b })
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			auto* a = this->inputs[0];
			auto* b = this->inputs[1];

			if (a->RequiresGrad()) {
				a->AccumulateGrad(gradOutput);
				a->Backward(gradOutput);
			}

			if (b->RequiresGrad()) {
				TensorCore::Tensor<T> negOutput = Operations::Negate(gradOutput, gradOutput.GetAllocator());

				b->AccumulateGrad(negOutput);
				b->Backward(negOutput);
			}
		}
	};
}