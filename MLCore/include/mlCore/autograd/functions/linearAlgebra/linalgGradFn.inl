 /// linalgGradFn.inl
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/linearAlgebra/linalg.h>

namespace MLCore::AutoGrad {
	template <typename T>
	DotGradFn<T>::DotGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b)
		: GradFn<T>({ a, b })
	{}
	
	template <typename T>
	void DotGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput){
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: DotGradFn: gradOutput shape mismatch");
		}

		TensorCore::Tensor<T> a{this->inputs[0]};
		TensorCore::Tensor<T> b{this->inputs[1]};

		if (a.NumElements() != b.NumElements()) {
			throw std::runtime_error("ERROR: DotGradFn: a and b shape mismatch");
		}

		T gradScalar = gradOutput[0];

		if (a.RequiresGrad()) {
			auto detachedB = b.Detach();
			auto gradA = Operations::MultiplyScalar(detachedB, gradScalar);
			a.Backward(gradA);
		}

		if (b.RequiresGrad()) {
			auto detachedA = a.Detach();
			auto gradB = Operations::MultiplyScalar(detachedA, gradScalar);
			b.Backward(gradB);
		}
	}
	
	template <typename T>
	MatMulGradFn<T>::MatMulGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b)
		: GradFn<T>({ a, b })
	{}
	
	template <typename T>
	void MatMulGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> a{this->inputs[0]};
		TensorCore::Tensor<T> b{this->inputs[1]};

		if (gradOutput.Rank() != 2 || gradOutput.Dims()[0] != a.Dims()[0] || gradOutput.Dims()[1] != b.Dims()[1]) {
			throw std::runtime_error("ERROR: MatMulGradFn: gradOutput shape mismatch");
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		if (a.RequiresGrad()) {
			auto detachedB = b.Detach();
			auto bT = Operations::Transpose(detachedB);

			auto gradA = Operations::MatMultiply(gradientOut, bT);
			a.Backward(gradA);
		}

		if (b.RequiresGrad()) {
			auto detachedA = a.Detach();
			auto aT = Operations::Transpose(detachedA);

			auto gradB = Operations::MatMultiply(aT, gradientOut);
			b.Backward(gradB);
		}
	}
	
	template <typename T>
	TransposeGradFn<T>::TransposeGradFn(std::shared_ptr<typename GradFn<T>::Impl> a)
		: GradFn<T>(a)
	{}
	
	template <typename T>
	void TransposeGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		if (gradOutput.Rank() != 2 || gradOutput.Dims()[0] != input.Dims()[1] || gradOutput.Dims()[1] != input.Dims()[0]) {
			throw std::runtime_error("ERROR: TransposeGradFn: gradOutput shape mismatch");
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		auto gradInput = Operations::Transpose(gradientOut);

		input.Backward(gradInput);
	}
}