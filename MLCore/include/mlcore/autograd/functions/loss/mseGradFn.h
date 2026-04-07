// mseGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class MSEGradFn : public GradFn<T> {
	public:
		MSEGradFn(TensorCore::Tensor<T>* pred, TensorCore::Tensor<T>* target)
			: GradFn<T>(pred), targetTensor(target)
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			if (gradOutput.NumElements() != 1) {
				throw std::runtime_error("ERROR: MSEGradFn: Expected a 1D tensor");
			}

			auto* predict = this->inputs[0];

			if (!predict->RequiresGrad()) {
				return;
			}

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			TensorCore::Tensor<T> gradInput{ predict->GetShape(), allocator};

			size_t size = gradInput.NumElements();

			T scale = static_cast<T>(2) / static_cast<T>(size);
			T gradScalar = gradOutput[0];

			for (size_t i = 0; i < size; ++i) {
				T diff = (*predict)[i] - (*targetTensor)[i];
				gradInput[i] = gradScalar * scale * diff;
			}

			predict->Backward(gradInput);
		}

	private:
		TensorCore::Tensor<T>* targetTensor;
	};
}