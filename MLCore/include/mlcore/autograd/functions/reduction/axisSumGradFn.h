// axisSumGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/operations/scalar/scalar.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class AxisSumGradFn : public GradFn<T> {
	public:
		AxisSumGradFn(TensorCore::Tensor<T>* a, size_t axis)
			: GradFn<T>(a), axis(axis), inputShape(a->GetShape()) {
			assert(axis < inputShape.Rank());
		}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			auto* input = this->inputs[0];

			if (!input->RequiresGrad()) {
				return;
			}

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			auto& gradOutputShape = gradOutput.GetShape();

			TensorCore::Tensor<T> gradInput{ inputShape, allocator };

			size_t size = gradInput.NumElements();

			for (size_t i = 0; i < size; ++i) {
				auto reduced = inputShape.UnflattenIndex(i);
				reduced.erase(reduced.begin() + axis);

				if (reduced.empty()) {
					reduced.push_back(0);
				}

				size_t gradOutputIndex = gradOutputShape.FlattenIndex(reduced);

				gradInput[i] = gradOutput[gradOutputIndex];
			}

			input->Backward(gradInput);
		}

	private:
		size_t axis;
		Utils::Shape inputShape;
	};
}