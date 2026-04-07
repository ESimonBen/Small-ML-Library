// maeGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class MAEGradFn : public GradFn<T> {
	public:
		MAEGradFn(TensorCore::Tensor<T>* pred, TensorCore::Tensor<T>* target)
			: GradFn<T>(pred), targetTensor(target)
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			if (gradOutput.NumElements() != 1) {
				throw std::runtime_error("ERROR: MAEGradFn: Expected a 1D tensor");
			}

			auto* predict = this->inputs[0];

			if (!predict->RequiresGrad()) {
				return;
			}

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			TensorCore::Tensor<T> gradInput{ predict->GetShape(), allocator };

			size_t size = gradInput.NumElements();

			T scale = static_cast<T>(1) / static_cast<T>(size);
			T gradScalar = gradOutput[0];

			for (size_t i = 0; i < size; ++i) {
				T diff = (*predict)[i] - (*targetTensor)[i];
				gradInput[i] = gradScalar * scale * static_cast<T>(((diff > 0) ? 1 : ((diff < 0) ? -1 : 0)));
			}

			predict->Backward(gradInput);
		}

	private:
		TensorCore::Tensor<T>* targetTensor;
	};
}