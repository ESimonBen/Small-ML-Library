// activation.inl
#include <cmath>
#include <algorithm>

namespace MLCore::Operations {
	template <typename T>
	TensorCore::Tensor<T> ReLU(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		for (size_t i = 0; i < A.NumElements(); ++i) {
			result[i] = std::max(T(0), A[i]);
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
				result[i] = T(1) / (T(1) + std::exp(-A[i]));
			}
			else {
				result[i] = std::exp(A[i]) / (T(1) + std::exp(A[i]));
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
}