// addGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/autograd/gradientUtils.h>

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
				auto gradA = ReduceSumToShape(gradOutput, a->GetShape());

				a->AccumulateGrad(gradA);

				if (a->HasGrad()) {
					a->Grad()->Backward(gradA);
				}
			}

			if (b->RequiresGrad()) {
				auto gradB = ReduceSumToShape(gradOutput, b->GetShape());

				b->AccumulateGrad(gradB);
				
				if (b->HasGrad()) {
					b->Grad()->Backward(gradB);
				}
			}
		}
	};
}