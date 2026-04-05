// elementwise.inl
#include <stdexcept>
#include <mlCore/operations/broadcast/broadcast.h>

namespace MLCore::Operations {
	template <typename T>
	inline TensorCore::Tensor<T> Add(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator) {
		if (!CanBroadcast(A.GetShape(), B.GetShape())) {
			throw std::runtime_error("ERROR: Tensor shapes cannot broadcast");
		}

		auto info = ComputeBroadcast(A.GetShape(), B.GetShape());
		TensorCore::Tensor<T> C{ info.shape, allocator };

		const size_t size = C.NumElements();

		if (A.GetShape() == B.GetShape()) {
			for (size_t i = 0; i < size; ++i) {
				C[i] = A[i] + B[i];
			}

			return C;
		}


		for (size_t i = 0; i < size; ++i) {
			size_t idxA = 0, idxB = 0, tmp = i;

			for (size_t j = 0; j < info.shape.Rank(); ++j) {
				size_t dimIndex = tmp / info.shape.Strides()[j];
				tmp %= info.shape.Strides()[j];

				idxA += dimIndex * info.strideA[j];
				idxB += dimIndex * info.strideB[j];
			}

			C[i] = A[idxA] + B[idxB];
		}

		return C;
	}

	template <typename T>
	inline TensorCore::Tensor<T> Subtract(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator) {
		if (!CanBroadcast(A.GetShape(), B.GetShape())) {
			throw std::runtime_error("ERROR: Tensor shapes cannot broadcast");
		}

		auto info = ComputeBroadcast(A.GetShape(), B.GetShape());
		TensorCore::Tensor<T> C{ info.shape, allocator };

		const size_t size = C.NumElements();

		if (A.GetShape() == B.GetShape()) {
			for (size_t i = 0; i < size; ++i) {
				C[i] = A[i] - B[i];
			}

			return C;
		}

		for (size_t i = 0; i < size; ++i) {
			size_t idxA = 0, idxB = 0, tmp = i;

			for (size_t j = 0; j < info.shape.Rank(); ++j) {
				size_t dimIndex = tmp / info.shape.Strides()[j];
				tmp %= info.shape.Strides()[j];

				idxA += dimIndex * info.strideA[j];
				idxB += dimIndex * info.strideB[j];
			}

			C[i] = A[idxA] - B[idxB];
		}

		return C;
	}

	template <typename T>
	inline TensorCore::Tensor<T> Multiply(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator) {
		if (!CanBroadcast(A.GetShape(), B.GetShape())) {
			throw std::runtime_error("ERROR: Tensor shapes cannot broadcast");
		}

		auto info = ComputeBroadcast(A.GetShape(), B.GetShape());
		TensorCore::Tensor<T> C{ info.shape, allocator };

		const size_t size = C.NumElements();

		if (A.GetShape() == B.GetShape()) {
			for (size_t i = 0; i < size; ++i) {
				C[i] = A[i] * B[i];
			}

			return C;
		}

		for (size_t i = 0; i < size; ++i) {
			size_t idxA = 0, idxB = 0, tmp = i;

			for (size_t j = 0; j < info.shape.Rank(); ++j) {
				size_t dimIndex = tmp / info.shape.Strides()[j];
				tmp %= info.shape.Strides()[j];

				idxA += dimIndex * info.strideA[j];
				idxB += dimIndex * info.strideB[j];
			}

			C[i] = A[idxA] * B[idxB];
		}

		return C;
	}

	template <typename T>
	inline TensorCore::Tensor<T> Divide(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator) {
		if (!CanBroadcast(A.GetShape(), B.GetShape())) {
			throw std::runtime_error("ERROR: Tensor shapes cannot broadcast");
		}

		auto info = ComputeBroadcast(A.GetShape(), B.GetShape());
		TensorCore::Tensor<T> C{ info.shape, allocator };

		const size_t size = C.NumElements();

		if (A.GetShape() == B.GetShape()) {
			for (size_t i = 0; i < size; ++i) {
				C[i] = A[i] / B[i];
			}

			return C;
		}

		for (size_t i = 0; i < size; ++i) {
			size_t idxA = 0, idxB = 0, tmp = i;

			for (size_t j = 0; j < info.shape.Rank(); ++j) {
				size_t dimIndex = tmp / info.shape.Strides()[j];
				tmp %= info.shape.Strides()[j];

				idxA += dimIndex * info.strideA[j];
				idxB += dimIndex * info.strideB[j];
			}

			C[i] = A[idxA] / B[idxB];
		}

		return C;
	}

	template <typename T>
	TensorCore::Tensor<T> Negate(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> B{ A.GetShape(), allocator };
		const size_t size = B.NumElements();

		for (size_t i = 0; i < size; ++i) {
			B[i] = -A[i];
		}

		return B;
	}

	template <typename T>
	TensorCore::Tensor<T> Square(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> B{ A.GetShape(), allocator };
		const size_t size = B.NumElements();

		for (size_t i = 0; i < size; ++i) {
			B[i] = A[i] * A[i];
		}

		return B;
	}
}