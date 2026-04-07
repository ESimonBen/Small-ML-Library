// squareGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/autograd/gradientUtils.h>
#include <mlCore/operations/scalar/scalar.h> 
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class SquareGradFn : public GradFn<T> {
	public:
		SquareGradFn(TensorCore::Tensor<T>* a)
			: GradFn<T>(a)
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			auto* input = this->inputs[0];

			if (!input->RequiresGrad()) {
				return;
			}

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			auto derivOfSquare = Operations::MultiplyScalarLeft(T(2), *input, allocator);

			auto gradInput = Operations::Multiply(gradOutput, derivOfSquare, allocator);

			input->Backward(gradInput);
		}
	};
}