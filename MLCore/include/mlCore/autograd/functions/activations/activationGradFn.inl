 /// activationGradFn.inl
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::AutoGrad {
	template <typename T>
	ReLUGradFn<T>::ReLUGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> a)
		: GradFn<T>(a)
	{}
	
	template <typename T>
	void ReLUGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (gradOutput.GetShape() != input.GetShape()) {
			throw std::runtime_error("Activation backward shape mismatch");
		}

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> gradInput{ input.GetShape(), input.GetAllocator() };


		for (size_t i = 0; i < gradInput.NumElements(); ++i) {
			gradInput[i] = (input[i] > static_cast<T>(0)) ? gradientOut[i] : static_cast<T>(0);
		}

		input.Backward(gradInput);
	}
	
	template <typename T>
	LeakyReLUGradFn<T>::LeakyReLUGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> a, T alpha)
		: GradFn<T>(a), alpha(alpha)
	{}
	
	template <typename T>
	void LeakyReLUGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (gradOutput.GetShape() != input.GetShape()) {
			throw std::runtime_error("Activation backward shape mismatch");
		}

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> gradInput{ input.GetShape(), input.GetAllocator() };

		size_t size = gradInput.NumElements();

		for (size_t i = 0; i < size; ++i) {
			gradInput[i] = (input[i] > static_cast<T>(0)) ? gradientOut[i] : alpha * gradientOut[i];
		}

		input.Backward(gradInput);
	}
	
	template <typename T>
	SoftmaxGradFn<T>::SoftmaxGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> a, std::shared_ptr<TensorCore::TensorImpl<T>> output)
		: GradFn<T>(a), m_OutputImpl(output)
	{}
	
	template <typename T>
	void SoftmaxGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		static_assert(std::is_floating_point_v<T>, "Softmax requires floating point type");

		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (gradOutput.GetShape() != input.GetShape()) {
			throw std::runtime_error("Activation backward shape mismatch");
		}

		if (!input.RequiresGrad()) {
			return;
		}

		auto lockedOutputImpl = m_OutputImpl.lock();

		if (!lockedOutputImpl) {
			throw std::runtime_error("ERROR: SoftmaxGradFn: Output tensor no longer exists");
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> output = TensorCore::Tensor<T>{ lockedOutputImpl }.Detach();

		size_t size = input.NumElements();

		T sum = 0;

		for (size_t i = 0; i < size; ++i) {
			sum += gradientOut[i] * output[i];
		}

		TensorCore::Tensor<T> gradInput = Operations::Multiply(output, Operations::SubtractScalar(gradientOut, sum, false));


		input.Backward(gradInput);
	}
	
	template <typename T>
	AxisSoftmaxGradFn<T>::AxisSoftmaxGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> a, std::shared_ptr<TensorCore::TensorImpl<T>> output, size_t axis)
		: GradFn<T>(a), m_OutputImpl(output), m_Axis(axis)
	{}
	
	template <typename T>
	void AxisSoftmaxGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (gradOutput.GetShape() != input.GetShape()) {
			throw std::runtime_error("Activation backward shape mismatch");
		}

		if (!input.RequiresGrad()) {
			return;
		}

		auto lockedOutputImpl = m_OutputImpl.lock();

		if (!lockedOutputImpl) {
			throw std::runtime_error("ERROR: AxisSoftmaxGradFn: Output tensor no longer exists");
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> y = TensorCore::Tensor<T>{ lockedOutputImpl }.Detach();

		TensorCore::Tensor<T> gradInput{ input.GetShape(), input.GetAllocator() };
		gradInput.Fill(static_cast<T>(0));

		const std::vector<size_t>& dims = input.Dims();
		size_t rank = input.Rank();

		/// Outer and inner size calculation
		size_t outer = 1;
		for (size_t i = 0; i < m_Axis; ++i) {
			outer *= dims[i];
		}

		size_t inner = 1;
		for (size_t i = m_Axis + 1; i < rank; ++i) {
			inner *= dims[i];
		}

		size_t axisSize = dims[m_Axis];

		for (size_t o = 0; o < outer; ++o) {
			for (size_t i = 0; i < inner; ++i) {
				size_t base = o * axisSize * inner + i;

				T dot = static_cast<T>(0);

				for (size_t j = 0; j < axisSize; ++j) {
					size_t idx = base + j * inner;
					dot += gradientOut[idx] * y[idx];
				}

				for (size_t j = 0; j < axisSize; ++j) {
					size_t idx = base + j * inner;
					gradInput[idx] += y[idx] * (gradientOut[idx] - dot);
				}
			}
		}

		input.Backward(gradInput);
	}
}
