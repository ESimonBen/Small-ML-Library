// gradientFn.h
#pragma once
#include <memory>
#include <vector>

// To avoid cirular dependencies
namespace MLCore::TensorCore {
	template <typename T>
	class Tensor;
}

namespace MLCore::AutoGrad {
	template <typename T>
	class GradFn {
	public:
		using Impl = TensorCore::Tensor<T>::Impl;

		GradFn() = default;

		explicit GradFn(std::shared_ptr<Impl> impl);
		explicit GradFn(std::vector<std::shared_ptr<Impl>> gradInput);

		virtual void Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) = 0;

		virtual ~GradFn() = default;

	protected:
		std::shared_ptr<Impl> Input(size_t i) {
			return inputs[i];
		}

		const std::vector<std::shared_ptr<Impl>>& Inputs() const {
			return inputs;
		}

		void PropogateToInput(size_t index, const TensorCore::Tensor<T>& grad) {
			TensorCore::Tensor<T> input{ inputs[index] };
			input.Backward(grad);
		}

	protected:
		std::vector<std::shared_ptr<Impl>> inputs;

		// May add something to store outputs at some point
	};
}

#include "gradientFn.inl"