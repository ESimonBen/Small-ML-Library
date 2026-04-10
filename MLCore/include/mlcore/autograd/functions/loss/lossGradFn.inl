// lossGradFn.inl
#include <algorithm>

namespace MLCore::AutoGrad {
	template <typename T>
	MSEGradFn<T>::MSEGradFn(TensorCore::Tensor<T>* pred, TensorCore::Tensor<T>* target)
		: GradFn<T>(pred), targetTensor(target)
	{}

	template <typename T>
	void MSEGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: MSEGradFn: Expected a 1D tensor");
		}

		auto* predict = this->inputs[0];

		if (!predict->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		TensorCore::Tensor<T> gradInput{ predict->GetShape(), allocator };

		size_t size = gradInput.NumElements();

		T scale = static_cast<T>(2) / static_cast<T>(size);
		T gradScalar = gradOutput[0];

		for (size_t i = 0; i < size; ++i) {
			T diff = (*predict)[i] - (*targetTensor)[i];
			gradInput[i] = gradScalar * scale * diff;
		}

		predict->Backward(gradInput);
	}

	template <typename T>
	MAEGradFn<T>::MAEGradFn(TensorCore::Tensor<T>* pred, TensorCore::Tensor<T>* target)
		: GradFn<T>(pred), targetTensor(target)
	{}

	template <typename T>
	void MAEGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: MAEGradFn: Expected a 1D tensor");
		}

		auto* predict = this->inputs[0];

		if (!predict->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		TensorCore::Tensor<T> gradInput{ predict->GetShape(), allocator };

		size_t size = gradInput.NumElements();

		T scale = static_cast<T>(1) / static_cast<T>(size);
		T gradScalar = gradOutput[0];

		for (size_t i = 0; i < size; ++i) {
			T diff = (*predict)[i] - (*targetTensor)[i];
			gradInput[i] = gradScalar * scale * static_cast<T>(((diff > 0) ? 1 : ((diff < 0) ? -1 : 0)));
		}

		predict->Backward(gradInput);
	}

	template <typename T>
	BCEGradFn<T>::BCEGradFn(TensorCore::Tensor<T>* pred, TensorCore::Tensor<T>* target)
		: GradFn<T>(pred), targetTensor(target)
	{}

	template <typename T>
	void BCEGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: BCEGradFn: Expected a 1D tensor");
		}

		auto* predict = this->inputs[0];

		if (!predict->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		TensorCore::Tensor<T> gradInput{ predict->GetShape(), allocator };

		size_t size = gradInput.NumElements();

		T gradScalar = gradOutput[0];
		T epsilon = static_cast<T>(1e-7);

		// May deal with batching in the next improvement (rather than a single output)
		for (size_t i = 0; i < size; ++i) {
			T p = (*predict)[i];
			p = std::clamp(p, epsilon, static_cast<T>(1) - epsilon);
			T t = (*targetTensor)[i];

			gradInput[i] = gradScalar * (-t / p + (static_cast<T>(1) - t) / (static_cast<T>(1) - p));
		}

		predict->Backward(gradInput);
	}

	template <typename T>
	CEGradFn<T>::CEGradFn(TensorCore::Tensor<T>* pred, TensorCore::Tensor<T>* target)
		: GradFn<T>(pred), targetTensor(target)
	{}

	template <typename T>
	void CEGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: CEGradFn: Expected a 1D tensor");
		}

		auto* predict = this->inputs[0];

		if (!predict->RequiresGrad()) {
			return;
		}

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		TensorCore::Tensor<T> gradInput{ predict->GetShape(), allocator };

		size_t size = gradInput.NumElements();

		T gradScalar = gradOutput[0];
		T epsilon = static_cast<T>(1e-7);

		// May deal with batching in the next improvement (rather than a single output)
		for (size_t i = 0; i < size; ++i) {
			T p = (*predict)[i];
			p = std::clamp(p, epsilon, static_cast<T>(1) - epsilon);
			T t = (*targetTensor)[i];

			gradInput[i] = gradScalar * (-t / p);
		}

		predict->Backward(gradInput);
	}
}