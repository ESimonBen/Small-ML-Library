// activationGradFn.inl
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::AutoGrad {
	template <typename T>
	ReLUGradFn<T>::ReLUGradFn(std::shared_ptr<typename GradFn<T>::Impl> a)
		: GradFn<T>(a)
	{
	}

	template <typename T>
	void ReLUGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (gradOutput.GetShape() != input.GetShape()) {
			throw std::runtime_error("Activation backward shape mismatch");
		}

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
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
	{
	}

	template <typename T>
	void LeakyReLUGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (gradOutput.GetShape() != input.GetShape()) {
			throw std::runtime_error("Activation backward shape mismatch");
		}

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> gradInput{ input.GetShape(), allocator };

		size_t size = gradInput.NumElements();

		for (size_t i = 0; i < size; ++i) {
			gradInput[i] = (input[i] > static_cast<T>(0)) ? gradientOut[i] : alpha * gradientOut[i];
		}

		input.Backward(gradInput);
	}

	template <typename T>
	SoftmaxGradFn<T>::SoftmaxGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b)
		: GradFn<T>(a), outputImpl(b)
	{
	}

	template <typename T>
	void SoftmaxGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) {
		static_assert(std::is_floating_point_v<T>, "Softmax requires floating point type");

		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (gradOutput.GetShape() != input.GetShape()) {
			throw std::runtime_error("Activation backward shape mismatch");
		}

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> output = TensorCore::Tensor<T>{ outputImpl }.Detach();

		size_t size = input.NumElements();

		T sum = 0;

		for (size_t i = 0; i < size; ++i) {
			sum += gradientOut[i] * output[i];
		}

		TensorCore::Tensor<T> gradInput = Operations::Multiply(output, Operations::SubtractScalar(gradientOut, sum, allocator, false), allocator);


		input.Backward(gradInput);
	}

	template <typename T>
	AxisSoftmaxGradFn<T>::AxisSoftmaxGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b, size_t axis)
		: GradFn<T>(a), outputImpl(b), axis(axis)
	{
	}

	template <typename T>
	void AxisSoftmaxGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> y = TensorCore::Tensor<T>{ outputImpl }.Detach();

		TensorCore::Tensor<T> gradInput{ input.GetShape(), allocator };

		const std::vector<size_t>& dims = input.Dims();
		size_t rank = input.Rank();

		// Outer and inner size calculation
		size_t outer = 1;
		for (size_t i = 0; i < axis; ++i) {
			outer *= dims[i];
		}

		size_t inner = 1;
		for (size_t i = axis + 1; i < rank; ++i) {
			inner *= dims[i];
		}

		size_t axisSize = dims[axis];

		for (size_t o = 0; o < outer; ++o) {
			for (size_t i = 0; i < inner; ++i) {
				size_t base = o * axisSize * inner + i;

				T dot = static_cast<T>(0);

				for (size_t j = 0; j < axisSize; ++j) {
					T idx = base + j * inner;
					dot += gradientOut[idx] * y[idx];
				}

				for (size_t j = 0; j < axisSize; ++j) {
					T idx = base + j * inner;
					gradInput[idx] += y[idx] * (gradientOut[idx] - dot);
				}
			}
		}

		input.Backward(gradInput);
	}
}
