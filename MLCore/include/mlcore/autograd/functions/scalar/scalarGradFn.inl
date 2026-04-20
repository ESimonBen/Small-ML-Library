// scalarGradFn.inl
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::AutoGrad {
	template <typename T>
	inline AddScalarGradFn<T>::AddScalarGradFn(std::shared_ptr<typename GradFn<T>::Impl> a)
		: GradFn<T>(a)
	{}

	template <typename T>
	void AddScalarGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input{this->inputs[0]};

		if (!input.RequiresGrad()) {
			return;
		}

		input.Backward(gradOutput);
	}

	template <typename T>
	SubScalarGradFn<T>::SubScalarGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, bool scalarOnLeft)
		: GradFn<T>(a), scalarOnLeft(scalarOnLeft)
	{}

	template <typename T>
	void SubScalarGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput)  {
		TensorCore::Tensor<T> input{this->inputs[0]};

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		TensorCore::Tensor<T> gradInput = (scalarOnLeft) ? Operations::Negate(gradientOut, allocator) : gradientOut;

		input.Backward(gradInput);
	}

	template <typename T>
	MulScalarGradFn<T>::MulScalarGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T scalar)
		: GradFn<T>(a), scalar(scalar)
	{}

	template <typename T>
	void MulScalarGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input{this->inputs[0]};

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> grad = gradOutput.Detach();
		auto& allocator = grad.GetAllocator();
		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		TensorCore::Tensor<T> gradInput = Operations::MultiplyScalar(gradientOut, scalar, allocator);

		input.Backward(gradInput);
	}

	template <typename T>
	DivScalarGradFn<T>::DivScalarGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T scalar, bool scalarOnLeft)
		: GradFn<T>(a), scalar(scalar), scalarOnLeft(scalarOnLeft)
	{}

	template <typename T>
	void DivScalarGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input{this->inputs[0]};

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		// Must create a detached version of the input to make sure another
		// computation graph is not created while backpropogating
		auto detachedInput = input.Detach();

		TensorCore::Tensor<T> gradInput = (scalarOnLeft) ?
			Operations::Multiply(gradientOut, Operations::DivideScalar(Operations::Square(detachedInput, allocator), -scalar, allocator, true), allocator)
			: Operations::DivideScalar(gradientOut, scalar, allocator, false);

		input.Backward(gradInput);
	}
}