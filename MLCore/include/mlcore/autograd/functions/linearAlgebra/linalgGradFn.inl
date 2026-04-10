// linalgGradFn.inl
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/linearAlgebra/linalg.h>

namespace MLCore::AutoGrad {
	template <typename T>
	DotGradFn<T>::DotGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
		: GradFn<T>({ a, b })
	{}
	
	template <typename T>
	void DotGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput){
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
			auto detachedB = b->Detach();
			auto gradA = Operations::MultiplyScalarLeft(gradScalar, detachedB, allocator);
			a->Backward(gradA);
		}

		if (b->RequiresGrad()) {
			auto detachedA = a->Detach();
			auto gradB = Operations::MultiplyScalarLeft(gradScalar, detachedA, allocator);
			b->Backward(gradB);
		}
	}

	template <typename T>
	MatMulGradFn<T>::MatMulGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
		: GradFn<T>({ a, b })
	{}

	template <typename T>
	void MatMulGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* a = this->inputs[0];
		auto* b = this->inputs[1];

		if (gradOutput.Dims() != std::vector<size_t>{a->Dims()[0], b->Dims()[1]}) {
			throw std::runtime_error("ERROR: MatMulGradFn: gradOutput shape mismatch");
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		if (a->RequiresGrad()) {
			auto detachedB = b->Detach();
			auto bT = Operations::Transpose(detachedB, allocator);

			auto gradA = Operations::MatMultiply(gradOutput, bT, allocator);
			a->Backward(gradA);
		}

		if (b->RequiresGrad()) {
			auto detachedA = a->Detach();
			auto aT = Operations::Transpose(detachedA, allocator);

			auto gradB = Operations::MatMultiply(aT, gradOutput, allocator);
			b->Backward(gradB);
		}
	}

	template <typename T>
	TransposeGradFn<T>::TransposeGradFn(TensorCore::Tensor<T>* a)
		: GradFn<T>(a)
	{}

	template <typename T>
	void TransposeGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		if (gradOutput.Dims()[0] != input->Dims()[1] || gradOutput.Dims()[1] != input->Dims()[0]) {
			throw std::runtime_error("ERROR: TransposeGradFn: gradOutput shape mismatch");
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		auto gradInput = Operations::Transpose(gradOutput, allocator);

		input->Backward(gradInput);
	}
}