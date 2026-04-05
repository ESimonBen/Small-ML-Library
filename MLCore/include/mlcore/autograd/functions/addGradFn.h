// addGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class AddGradFn : public GradFn<T> {
	public:
		AddGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
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
				b->AccumulateGrad(gradOutput);
				b->Backward(gradOutput);
			}
		}
	};
}