// dotGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/operations/scalar/scalar.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class DotGradFn : public GradFn<T> {
	public:
		DotGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
			: GradFn<T>({ a, b })
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			if (gradOutput.NumElements() != 1) {
				throw std::runtime_error("ERROR: DotGradFn: gradOutput shape mismatch");
			}

			auto* a = this->inputs[0];
			auto* b = this->inputs[1];

			if (a->GetShape() != b->GetShape()) {
				throw std::runtime_error("ERROR: DotGradFn: a and b shape mismatch")
			}

			T gradScalar = gradOutput[0];

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			if (a->RequiresGrad()) {
				auto gradA = Operations::MultiplyScalarLeft(gradScalar, *b, allocator);
				a->Backward(gradA);
			}

			if (b->RequiresGrad()) {
				auto gradB = Operations::MultiplyScalarLeft(gradScalar, *a, allocator);
				b->Backward(gradB);
			}
		}
	};
}