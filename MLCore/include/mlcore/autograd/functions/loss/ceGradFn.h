// ceGradFn.h
#pragma once
#include <algorithm>
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class CEGradFn : public GradFn<T> {
	public:
		CEGradFn(TensorCore::Tensor<T>* pred, TensorCore::Tensor<T>* target)
			: GradFn<T>(pred), targetTensor(target)
		{}

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput) override {
			if (gradOutput.NumElements() != 1) {
				throw std::runtime_error("ERROR: CEGradFn: Expected a 1D tensor");
			}

			auto* predict = this->inputs[0];

			if (!predict->RequiresGrad()) {
				return;
			}

			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

			TensorCore::Tensor<T> gradInput{ predict->GetShape(), allocator };

			size_t size = gradInput.NumElements();

			T gradScalar = gradOutput[0];
			T epsilon = static_cast<T>(1e-7);

			// May deal with batching in the next improvement (rather than a single output)
			for (size_t i = 0; i < size; ++i) {
				T p = (*predict)[i];
				p = std::clamp(p, epsilon, static_cast<T>(1) - epsilon);
				T t = (*targetTensor)[i];

				gradInput[i] = gradScalar * (-t / p);
			}

			predict->Backward(gradInput);
		}

	private:
		TensorCore::Tensor<T>* targetTensor;
	};
}