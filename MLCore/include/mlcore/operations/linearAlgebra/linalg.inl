// linalg.inl
#include <vector>
#include <stdexcept>

namespace MLCore::Operations {
	template <typename T>
	TensorCore::Tensor<T> MatMultiply(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator) {
		if (A.Rank() != 2 || B.Rank() != 2) {
			throw std::runtime_error("ERROR: MatMultiply: Only 2D tensors supported for now");
		}

		size_t M = A.Dims()[0];
		size_t K = A.Dims()[1];
		size_t N = B.Dims()[1];

		if (K != B.Dims()[0]) {
			throw std::runtime_error("ERROR: MatMultiply: Inner dimensions do not match");
		}

		TensorCore::Tensor<T> C{ {M, N}, allocator };

		for (size_t i = 0; i < M; ++i) {
			for (size_t j = 0; j < N; ++j) {
				T sum = T{};
				for (size_t k = 0; k < K; ++k) {
					sum += A[i * K + k] * B[k * N + j];
				}
				C[i * N + j] = sum;
			}
		}

		return C;
	}

	template <typename T>
	TensorCore::Tensor<T> Transpose(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		if (A.Rank() != 2) {
			throw std::runtime_error("ERROR: Transpose: Only 2D tensors supported");
		}

		size_t M = A.Dims()[0];
		size_t N = A.Dims()[1];

		TensorCore::Tensor<T> B{ {N, M}, allocator };

		for (size_t i = 0; i < M; ++i) {
			for (size_t j = 0; j < N; ++j) {
				B[j * M + i] = A[i * N + j];
			}
		}

		return B;
	}

	template <typename T>
	TensorCore::Tensor<T> Dot(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator) {
		if (A.Rank() != 1 || B.Rank() != 1 || A.NumElements() != B.NumElements()) {
			throw std::runtime_error("ERROR: Dot: Only 1D tensors of the same size supported");
		}

		T sum = T{};

		for (size_t i = 0; i < A.NumElements(); ++i) {
			sum += A[i] * B[i];
		}

		TensorCore::Tensor<T> C{ {1}, allocator };
		C[0] = sum;

		return C;
	}
}