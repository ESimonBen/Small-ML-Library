// scalarGradFn.inl
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::AutoGrad {
	template <typename T>
	inline AddScalarGradFn<T>::AddScalarGradFn(TensorCore::Tensor<T>* a)
		: GradFn<T>(a)
	{}

	template <typename T>
	void AddScalarGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		input->Backward(gradOutput);
	}

	template <typename T>
	SubScalarGradFn<T>::SubScalarGradFn(TensorCore::Tensor<T>* a, bool scalarOnLeft)
		: GradFn<T>(a), scalarOnLeft(scalarOnLeft)
	{}

	template <typename T>
	void SubScalarGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput)  {
		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		TensorCore::Tensor<T> gradInput = (scalarOnLeft) ? Operations::Negate(gradOutput, allocator) : gradOutput;

		input->Backward(gradInput);
	}

	template <typename T>
	MulScalarGradFn<T>::MulScalarGradFn(TensorCore::Tensor<T>* a, T scalar)
		: GradFn<T>(a), scalar(scalar)
	{}

	template <typename T>
	void MulScalarGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		TensorCore::Tensor<T> gradInput = Operations::MultiplyScalar(gradOutput, scalar, allocator);

		input->Backward(gradInput);
	}

	template <typename T>
	DivScalarGradFn<T>::DivScalarGradFn(TensorCore::Tensor<T>* a, T scalar, bool scalarOnLeft)
		: GradFn<T>(a), scalar(scalar), scalarOnLeft(scalarOnLeft)
	{}

	template <typename T>
	void DivScalarGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		TensorCore::Tensor<T> gradInput = (scalarOnLeft) ?
			Operations::Multiply(gradOutput, Operations::DivideScalarLeft(-scalar, Operations::Square(*input, allocator), allocator), allocator)
			: Operations::DivideScalarRight(gradOutput, scalar, allocator);

		input->Backward(gradInput);
	}
}