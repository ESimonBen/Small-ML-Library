 /// scalarGradFn.inl
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

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		input.Backward(gradientOut);
	}
	
	template <typename T>
	SubScalarGradFn<T>::SubScalarGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, bool scalarOnLeft)
		: GradFn<T>(a), m_ScalarOnLeft(scalarOnLeft)
	{}
	
	template <typename T>
	void SubScalarGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput)  {
		TensorCore::Tensor<T> input{this->inputs[0]};

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		TensorCore::Tensor<T> gradInput = (m_ScalarOnLeft) ? Operations::Negate(gradientOut) : gradientOut;

		input.Backward(gradInput);
	}
	
	template <typename T>
	MulScalarGradFn<T>::MulScalarGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T scalar)
		: GradFn<T>(a), m_Scalar(scalar)
	{}
	
	template <typename T>
	void MulScalarGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input{this->inputs[0]};

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		TensorCore::Tensor<T> gradInput = Operations::MultiplyScalar(gradientOut, m_Scalar);

		input.Backward(gradInput);
	}
	
	template <typename T>
	DivScalarGradFn<T>::DivScalarGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T scalar, bool scalarOnLeft)
		: GradFn<T>(a), m_Scalar(scalar), m_ScalarOnLeft(scalarOnLeft)
	{}
	
	template <typename T>
	void DivScalarGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input{this->inputs[0]};

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		

		/// Must create a detached version of the input to make sure another computation graph is not created while backpropogating
		auto detachedInput = input.Detach();

		TensorCore::Tensor<T> gradInput = (m_ScalarOnLeft) ?
			Operations::Multiply(gradientOut, Operations::DivideScalar(Operations::Square(detachedInput), -m_Scalar, true))
			: Operations::DivideScalar(gradientOut, m_Scalar, false);

		input.Backward(gradInput);
	}
}