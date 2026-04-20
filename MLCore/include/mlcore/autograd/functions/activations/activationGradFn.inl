// activationGradFn.inl
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::AutoGrad {
	template <typename T>
	ReLUGradFn<T>::ReLUGradFn(std::shared_ptr<typename GradFn<T>::Impl> a)
		: GradFn<T>(a)
	{}

	template <typename T>
	void ReLUGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input {this->inputs[0]};

		if (gradOutput.GetShape() != input.GetShape()) {
			throw std::runtime_error("Activation backward shape mismatch");
		}

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		TensorCore::Tensor<T> gradInput{ input.GetShape(), allocator };


		// Want to replace this with a more readable version (replace static_cast<T>(0) with Tensor::Zero or something)
		for (size_t i = 0; i < gradInput.NumElements(); ++i) {
			gradInput[i] = (input[i] > static_cast<T>(0)) ? gradientOut[i] : static_cast<T>(0);
		}

		input.Backward(gradInput);
	}

	template <typename T>
	LeakyReLUGradFn<T>::LeakyReLUGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T alpha)
		: GradFn<T>(a), alpha(alpha)
	{}

	template <typename T>
	void LeakyReLUGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input {this->inputs[0]};

		if (gradOutput.GetShape() != input.GetShape()) {
			throw std::runtime_error("Activation backward shape mismatch");
		}

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		TensorCore::Tensor<T> gradInput{ input.GetShape(), allocator };

		size_t size = gradInput.NumElements();

		for (size_t i = 0; i < size; ++i) {
			gradInput[i] = (input[i] > static_cast<T>(0)) ? gradientOut[i] : alpha * gradientOut[i];
		}

		input.Backward(gradInput);
	}

	template <typename T>
	SigmoidGradFn<T>::SigmoidGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b)
		: GradFn<T>(a), outputImpl(b)
	{}

	template <typename T>
	void SigmoidGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		static_assert(std::is_floating_point_v<T>, "Sigmoid requires floating point type");

		TensorCore::Tensor<T> input {this->inputs[0]};

		if (gradOutput.GetShape() != input.GetShape()) {
			throw std::runtime_error("Activation backward shape mismatch");
		}

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		TensorCore::Tensor<T> gradInput{ input.GetShape(), allocator };
		TensorCore::Tensor<T> output = TensorCore::Tensor<T>{outputImpl}.Detach();


		size_t size = gradInput.NumElements();

		for (size_t i = 0; i < size; ++i) {
			T sigX = output[i];
			
			gradInput[i] = gradientOut[i] * sigX * (static_cast<T>(1) - sigX);
		}

		input.Backward(gradInput);
	}

	template <typename T>
	TanhGradFn<T>::TanhGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b)
		: GradFn<T>(a), outputImpl(b)
	{}

	template <typename T>
	void TanhGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		static_assert(std::is_floating_point_v<T>, "Tanh requires floating point type");

		TensorCore::Tensor<T> input {this->inputs[0]};

		if (gradOutput.GetShape() != input.GetShape()) {
			throw std::runtime_error("Activation backward shape mismatch");
		}

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		TensorCore::Tensor<T> gradInput{ input.GetShape(), allocator };
		TensorCore::Tensor<T> output = TensorCore::Tensor<T>{ outputImpl }.Detach();


		size_t size = gradInput.NumElements();

		for (size_t i = 0; i < size; ++i) {
			T tanhX = output[i];

			gradInput[i] = gradientOut[i] * (static_cast<T>(1) - (tanhX * tanhX));
		}

		input.Backward(gradInput);
	}

	template <typename T>
	SoftmaxGradFn<T>::SoftmaxGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b)
		: GradFn<T>(a), outputImpl(b)
	{}

	template <typename T>
	void SoftmaxGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		static_assert(std::is_floating_point_v<T>, "Softmax requires floating point type");

		TensorCore::Tensor<T> input {this->inputs[0]};

		if (gradOutput.GetShape() != input.GetShape()) {
			throw std::runtime_error("Activation backward shape mismatch");
		}

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		TensorCore::Tensor<T> output = TensorCore::Tensor<T>{outputImpl}.Detach();

		size_t size = input.NumElements();

		T sum = 0;

		for (size_t i = 0; i < size; ++i) {
			sum += gradientOut[i] * output[i];
		}

		TensorCore::Tensor<T> gradInput = Operations::Multiply(output, Operations::SubtractScalar(gradientOut, sum, allocator, false), allocator);


		input.Backward(gradInput);
	}
}