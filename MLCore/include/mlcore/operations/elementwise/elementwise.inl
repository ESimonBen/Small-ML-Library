// elementwise.inl
#include <cmath>
#include <stdexcept>
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/broadcast/broadcast.h>
#include <mlCore/autograd/functions/elementwise/elementwiseGradFn.h>

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
		}
		else {
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
		}

		if (A.RequiresGrad() || B.RequiresGrad()) {
			C.SetRequiresGrad(true);
			C.SetGradFn(std::make_shared<AutoGrad::AddGradFn<T>>(A.GetImpl(), B.GetImpl()));
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
		}
		else {
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
		}

		if (A.RequiresGrad() || B.RequiresGrad()) {
			C.SetRequiresGrad(true);
			C.SetGradFn(std::make_shared<AutoGrad::SubGradFn<T>>(A.GetImpl(), B.GetImpl()));
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
		}
		else {
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
		}

		if (A.RequiresGrad() || B.RequiresGrad()) {
			C.SetRequiresGrad(true);
			C.SetGradFn(std::make_shared<AutoGrad::MulGradFn<T>>(A.GetImpl(), B.GetImpl()));
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
		}
		else {
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
		}

		if (A.RequiresGrad() || B.RequiresGrad()) {
			C.SetRequiresGrad(true);
			C.SetGradFn(std::make_shared<AutoGrad::DivGradFn<T>>(A.GetImpl(), B.GetImpl()));
		}

		return C;
	}

	template <typename T>
	[[nodiscard]] TensorCore::Tensor<T> Power(const TensorCore::Tensor<T>& A, T exponent, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> B{ A.GetShape(), allocator };
		const size_t size = B.NumElements();

		for (size_t i = 0; i < size; ++i) {
			B[i] = std::pow(A[i], exponent); // May be optimized in the future
		}

		if (A.RequiresGrad()) {
			B.SetRequiresGrad(true);
			B.SetGradFn(std::make_shared<AutoGrad::PowerGradFn<T>>(A.GetImpl(), exponent));;
		}

		return B;
	}

	template <typename T>
	TensorCore::Tensor<T> Abs(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> B{ A.GetShape(), allocator };
		size_t size = B.NumElements();

		for (size_t i = 0; i < size; ++i) {
			T num = A[i];
			B[i] = (num < 0) ? -num : num; // Automatically catches the 0 case, because 0 is not less than 0, so the value will be 0
		}

		if (A.RequiresGrad()) {
			B.SetRequiresGrad(true);
			B.SetGradFn(std::make_shared<AutoGrad::AbsGradFn<T>>(A.GetImpl()));
		}

		return B;
	}

	template <typename T>
	TensorCore::Tensor<T> Clamp(const TensorCore::Tensor<T>& A, T min, T max, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		size_t size = A.NumElements();

		for (size_t i = 0; i < size; ++i) {
			T inp = A[i];
			result[i] = std::min(std::max(inp, max), max);
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::ClampGradFn<T>>(A.GetImpl(), min, max));
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> Log(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		size_t size = A.NumElements();

		for (size_t i = 0; i < size; ++i) {
			result[i] = std::log(A[i]);
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::LogGradFn<T>>(A.GetImpl()));
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> Exp(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		size_t size = A.NumElements();

		for (size_t i = 0; i < size; ++i) {
			result[i] = std::exp(A[i]);
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::ExpGradFn<T>>(A.GetImpl()));
		}

		return result;
	}

	template<typename T>
	TensorCore::Tensor<T> Equal(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B, Memory::ArenaAllocator& allocator) {
		if (A.GetShape() != B.GetShape()) {
			throw std::runtime_error("ERROR: Equal: Tensors are not the same shape");
		}
		
		TensorCore::Tensor<T> C{ A.GetShape, allocator };

		size_t size = A.NumElements();

		for (size_t i = 0; i < size; ++i) {
			C[i] = (A[i] == B[i]) ? static_cast<T>(1) : static_cast<T>(0);
		}

		C.SetRequiresGrad(false); // No gradient function necessary

		return C;
	}

	template <typename T>
	TensorCore::Tensor<T> Negate(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> result = MultiplyScalar(A, static_cast<T>(-1), allocator);

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> Square(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> result = Power(A, static_cast<T>(2), allocator);

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> Reciprocal(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> result = DivideScalar(A, static_cast<T>(1), allocator, true);

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
		}

		return result;
	}
}