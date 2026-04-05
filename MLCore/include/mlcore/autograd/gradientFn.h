// gradientFn.h
#pragma once
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
		using Tensor = TensorCore::Tensor<T>;

		GradFn() = default;

		explicit GradFn(Tensor* gradInput);

		explicit GradFn(const std::vector<Tensor*>& gradInput);

		virtual void Backward(const Tensor& gradOutput) = 0;

		virtual ~GradFn() = default;

	protected:
		std::vector<Tensor*> inputs;
	};
}

#include "gradientFn.inl"