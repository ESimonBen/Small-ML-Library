// activationGradFn.inl

namespace MLCore::AutoGrad {
	template <typename T>
	ReLUGradFn<T>::ReLUGradFn(TensorCore::Tensor<T>* a)
		: GradFn<T>(a)
	{}

	template <typename T>
	void ReLUGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		auto& gradOutputShape = gradOutput.GetShape();

		TensorCore::Tensor<T> gradInput{ input->GetShape(), allocator };


		// Want to replace this with a more readable version (replace static_cast<T>(0) with Tensor::Zero or something)
		for (size_t i = 0; i < gradInput.NumElements(); ++i) {
			gradInput[i] = ((*input)[i] > static_cast<T>(0)) ? gradOutput[i] : static_cast<T>(0);
		}

		input->Backward(gradInput);
	}

	template <typename T>
	LeakyReLUGradFn<T>::LeakyReLUGradFn(TensorCore::Tensor<T>* a, T alpha)
		: GradFn<T>(a), alpha(alpha)
	{}

	template <typename T>
	void LeakyReLUGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		TensorCore::Tensor<T> gradInput{ input->GetShape(), allocator };

		size_t size = gradInput.NumElements();

		for (size_t i = 0; i < size; ++i) {
			gradInput[i] = ((*input)[i] > static_cast<T>(0)) ? gradOutput[i] : alpha * gradOutput[i];
		}

		input->Backward(gradInput);
	}

	template <typename T>
	SigmoidGradFn<T>::SigmoidGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
		: GradFn<T>(a), outputTensor(b)
	{}

	template <typename T>
	void SigmoidGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		TensorCore::Tensor<T> gradInput{ input->GetShape(), allocator };

		size_t size = gradInput.NumElements();

		for (size_t i = 0; i < size; ++i) {
			T sigX = (*outputTensor)[i];

			gradInput[i] = gradOutput[i] * sigX * (static_cast<T>(1) - sigX);
		}

		input->Backward(gradInput);
	}

	template <typename T>
	TanhGradFn<T>::TanhGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
		: GradFn<T>(a), outputTensor(b)
	{}

	template <typename T>
	void TanhGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		TensorCore::Tensor<T> gradInput{ input->GetShape(), allocator };

		size_t size = gradInput.NumElements();

		for (size_t i = 0; i < size; ++i) {
			T tanhX = (*outputTensor)[i];

			gradInput[i] = gradOutput[i] * (static_cast<T>(1) - (tanhX * tanhX));
		}

		input->Backward(gradInput);
	}

	template <typename T>
	SoftmaxGradFn<T>::SoftmaxGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
		: GradFn<T>(a), outputTensor(b)
	{}

	template <typename T>
	void SoftmaxGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		TensorCore::Tensor<T> gradInput{ input->GetShape(), allocator };

		size_t size = gradInput.NumElements();

		T sum = 0;

		for (size_t i = 0; i < size; ++i) {
			sum += gradOutput[i] * (*outputTensor)[i];
		}

		// Should probably use scalar subtraction for this
		for (size_t i = 0; i < size; ++i) {
			gradInput[i] = (*outputTensor)[i] * (gradOutput[i] - sum);
		}

		input->Backward(gradInput);
	}
}