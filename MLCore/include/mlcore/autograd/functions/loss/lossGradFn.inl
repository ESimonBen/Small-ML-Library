// lossGradFn.inl
#include <algorithm>
#include <mlCore/operations/activations/activation.h>

namespace MLCore::AutoGrad {
	template <typename T>
	MSEGradFn<T>::MSEGradFn(std::shared_ptr<typename GradFn<T>::Impl> pred, std::shared_ptr<typename GradFn<T>::Impl> target)
		: GradFn<T>(pred), targetImpl(target)
	{}

	template <typename T>
	void MSEGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: MSEGradFn: Expected a scalar tensor");
		}

		TensorCore::Tensor<T> predict {this->inputs[0]};
		TensorCore::Tensor<T> targetTensor{ targetImpl };

		if (!predict.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		TensorCore::Tensor<T> gradInput{ predict.GetShape(), allocator };

		size_t size = gradInput.NumElements();

		T scale = static_cast<T>(2) / static_cast<T>(size);
		T gradScalar = gradOutput[0];

		for (size_t i = 0; i < size; ++i) {
			T diff = predict[i] - targetTensor[i];
			gradInput[i] = gradScalar * scale * diff;
		}

		predict.Backward(gradInput);
	}

	template <typename T>
	MAEGradFn<T>::MAEGradFn(std::shared_ptr<typename GradFn<T>::Impl> pred, std::shared_ptr<typename GradFn<T>::Impl> target)
		: GradFn<T>(pred), targetImpl(target)
	{}

	template <typename T>
	void MAEGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: MAEGradFn: Expected a scalar tensor");
		}

		TensorCore::Tensor<T> predict {this->inputs[0]};
		TensorCore::Tensor<T> targetTensor{ targetImpl };

		if (!predict.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		TensorCore::Tensor<T> gradInput{ predict.GetShape(), allocator };

		size_t size = gradInput.NumElements();

		T scale = static_cast<T>(1) / static_cast<T>(size);
		T gradScalar = gradOutput[0];

		for (size_t i = 0; i < size; ++i) {
			T diff = predict[i] - targetTensor[i];
			gradInput[i] = gradScalar * scale * static_cast<T>(((diff > 0) ? 1 : ((diff < 0) ? -1 : 0)));
		}

		predict.Backward(gradInput);
	}

	template <typename T>
	BCEGradFn<T>::BCEGradFn(std::shared_ptr<typename GradFn<T>::Impl> pred, std::shared_ptr<typename GradFn<T>::Impl> target)
		: GradFn<T>(pred), targetImpl(target)
	{}

	template <typename T>
	void BCEGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: BCEGradFn: Expected a scalar tensor");
		}

		TensorCore::Tensor<T> predict {this->inputs[0]};

		if (!predict.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		TensorCore::Tensor<T> gradInput{ predict.GetShape(), allocator };
		TensorCore::Tensor<T> targetTensor{ targetImpl };

		size_t size = gradInput.NumElements();

		T gradScalar = gradOutput[0];
		T epsilon = static_cast<T>(1e-7);

		// May deal with batching in the next improvement (rather than a single output)
		for (size_t i = 0; i < size; ++i) {
			T p = predict[i];
			p = std::clamp(p, epsilon, static_cast<T>(1) - epsilon);
			T t = targetTensor[i];

			gradInput[i] = gradScalar * (-t / p + (static_cast<T>(1) - t) / (static_cast<T>(1) - p));
		}

		predict.Backward(gradInput);
	}

	template <typename T>
	BCEWithLogitsGradFn<T>::BCEWithLogitsGradFn(std::shared_ptr<typename GradFn<T>::Impl> pred, std::shared_ptr<typename GradFn<T>::Impl> target)
		: GradFn<T>(pred), targetImpl(target)
	{}

	template <typename T>
	void BCEWithLogitsGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: BCEWithLogitsGradFn: expected scalar grad");
		}

		TensorCore::Tensor<T> logits{ this->inputs[0] };

		if (!logits.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> targets{ targetImpl };
		TensorCore::Tensor<T> detachedLogits = logits.Detach();
		auto& allocator = detachedLogits.GetAllocator();
		size_t size = logits.NumElements();

		TensorCore::Tensor<T> sigmoid = Operations::Sigmoid(detachedLogits, allocator);

		TensorCore::Tensor<T> gradInput{ logits.GetShape(), allocator };
		T gradScalar = gradOutput[0];

		for (size_t i = 0; i < size; ++i) {
			gradInput[i] = gradScalar * (static_cast<T>(1) / size) *(sigmoid[i] - targets[i]);
		}

		logits.Backward(gradInput);
	}

	template <typename T>
	CEGradFn<T>::CEGradFn(std::shared_ptr<typename GradFn<T>::Impl> pred, std::shared_ptr<typename GradFn<T>::Impl> target)
		: GradFn<T>(pred), targetImpl(target)
	{}

	template <typename T>
	void CEGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: CEGradFn: Expected a scalar tensor");
		}

		TensorCore::Tensor<T> predict {this->inputs[0]};

		if (!predict.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		TensorCore::Tensor<T> gradInput{ predict.GetShape(), allocator };
		TensorCore::Tensor<T> targetTensor{ targetImpl };

		size_t size = gradInput.NumElements();

		T gradScalar = gradOutput[0];
		T epsilon = static_cast<T>(1e-7);

		// May deal with batching in the next improvement (rather than a single output)
		for (size_t i = 0; i < size; ++i) {
			T p = predict[i];
			p = std::clamp(p, epsilon, static_cast<T>(1) - epsilon);
			T t = targetTensor[i];

			gradInput[i] = gradScalar * (static_cast<T>(1) / size) * (-t / p);
		}

		predict.Backward(gradInput);
	}

	template <typename T>
	CEWithLogitsGradFn<T>::CEWithLogitsGradFn(std::shared_ptr<typename GradFn<T>::Impl> pred, std::shared_ptr<typename GradFn<T>::Impl> target)
		: GradFn<T>(pred), targetImpl(target)
	{}

	template <typename T>
	void CEWithLogitsGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: CEWithLogitsGradFn: Expected a scalar tensor");
		}

		TensorCore::Tensor<T> logits{ this->inputs[0] };
		TensorCore::Tensor<T> target{ targetImpl };

		TensorCore::Tensor<T> detachedLogits = logits.Detach();
		auto& allocator = detachedLogits.GetAllocator();
		size_t size = logits.NumElements();

		TensorCore::Tensor<T> softmax = Operations::Softmax(detachedLogits, allocator);

		TensorCore::Tensor<T> gradInput{ logits.GetShape(), allocator };
		T gradScalar = gradOutput[0];

		for (size_t i = 0; i < size; ++i) {
			gradInput[i] = gradScalar * ((softmax[i] - target[i]) / size);
		}

		logits.Backward(gradInput);
	}
}