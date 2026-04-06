// matMulGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/operations/linearAlgebra/linalg.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class MatMulGradFn : public GradFn<T> {
	public:
		MatMulGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
			: GradFn<T>({ a, b })
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			auto* a = this->inputs[0];
			auto* b = this->inputs[1];

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			if (a->RequiresGrad()) {
				auto bT = Operations::Transpose((*b), allocator);

				auto gradA = Operations::MatMultiply(gradOutput, bT, allocator);

				a->AccumulateGrad(gradA);

				if (a->HasGrad()) {
					a->Grad()->Backward(gradA);
				}
			}

			if (b->RequiresGrad()) {
				auto aT = Operations::Transpose((*a), allocator);

				auto gradB = Operations::MatMultiply(aT, gradOutput, allocator);

				b->AccumulateGrad(gradB);
				
				if (b->HasGrad()) {
					b->Grad()->Backward(gradB);
				}
			}
		}
	};
}