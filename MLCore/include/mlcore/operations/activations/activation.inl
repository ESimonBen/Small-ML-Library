// activation.inl
#include <cmath>
#include <algorithm>

namespace MLCore::AutoGrad {
	template <typename T>
	class ReLUGradFn;
}

namespace MLCore::Operations {
	template <typename T>
	TensorCore::Tensor<T> ReLU(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		for (size_t i = 0; i < A.NumElements(); ++i) {
			result[i] = std::max(static_cast<T>(0), A[i]);
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(new AutoGrad::ReLUGradFn<T>(const_cast<TensorCore::Tensor<T>*>(&A)));
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> LeakyReLU(const TensorCore::Tensor<T>& A, T alpha, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		for (size_t i = 0; i < A.NumElements(); ++i) {
			result[i] = (A[i] > 0) ? A[i] : alpha * A[i];
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> Sigmoid(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		for (size_t i = 0; i < A.NumElements(); ++i) {
			if (A[i] >= 0) {
				result[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-A[i]));
			}
			else {
				result[i] = std::exp(A[i]) / (static_cast<T>(1) + std::exp(A[i]));
			}
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> Tanh(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		for (size_t i = 0; i < A.NumElements(); ++i) {
			result[i] = std::tanh(A[i]);
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> Softmax(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> result{ A.GetShape(), allocator };
		size_t size = A.NumElements();
		T maxValue = A[0];

		for (size_t i = 0; i < size; ++i) {
			if (A[i] > maxValue) {
				maxValue = A[i];
			}
		}

		T sumExp = static_cast<T>(0);

		for (size_t i = 0; i < size; ++i) {
			result[i] = std::exp(A[i] - maxValue);
			sumExp += result[i];
		}

		for (size_t i = 0; i < size; ++i) {
			result[i] /= sumExp;
		}

		return result;
	}
}