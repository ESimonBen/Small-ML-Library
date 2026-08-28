 /// linalg.inl
#include <vector>
#include <stdexcept>
#include <mlCore/autograd/functions/linearAlgebra/linalgGradFn.h>

namespace MLCore::Operations {
	template <typename T>
	inline TensorCore::Tensor<T> MatMultiply(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B) {
		if (&A.GetAllocator() != &B.GetAllocator()) {
			throw std::runtime_error("ERROR: Operations between tensors on different allocators are forbidden");
		}

		if (A.Rank() != 2 || B.Rank() != 2) {
			throw std::runtime_error("ERROR: MatMultiply: Only 2D tensors supported for now");
		}

		size_t M = A.Dims()[0];
		size_t K = A.Dims()[1];
		size_t N = B.Dims()[1];

		if (K != B.Dims()[0]) {
			throw std::runtime_error("ERROR: MatMultiply: Inner dimensions do not match");
		}

		Memory::ArenaAllocator& allocator = A.GetAllocator();
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

		if (A.RequiresGrad() || B.RequiresGrad()) {
			C.SetRequiresGrad(true);
			C.SetGradFn(std::make_shared<AutoGrad::MatMulGradFn<T>>(A.GetImpl(), B.GetImpl()));
		}

		return C;
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> Transpose(const TensorCore::Tensor<T>& A) {
		if (A.Rank() != 2) {
			throw std::runtime_error("ERROR: Transpose: Only 2D tensors supported");
		}

		std::vector<size_t> dims = { A.Dims()[1], A.Dims()[0] };

		std::vector<size_t> strides = { A.Strides()[1], A.Strides()[0] };

		auto impl = std::make_shared<TensorCore::TensorImpl<T>>(
			Utils::Shape{ dims },
			strides,
			A.GetImpl()->storage,
			A.GetImpl()->allocator,
			A.GetImpl()->offset,
			A.RequiresGrad(),
			nullptr,
			nullptr
		);

		TensorCore::Tensor<T> result{impl};

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::TransposeGradFn<T>>(A.GetImpl()));
		}

		return result;
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> Dot(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B) {
		if (&A.GetAllocator() != &B.GetAllocator()) {
			throw std::runtime_error("ERROR: Operations between tensors on different allocators are forbidden");
		}

		if (A.IsEmpty() || B.IsEmpty()) {
			throw std::runtime_error("ERROR: Input tensors cannot be null");
		}

		if (A.Rank() != 1 || B.Rank() != 1 || A.NumElements() != B.NumElements()) {
			throw std::runtime_error("ERROR: Dot: Only 1D tensors of the same size supported");
		}

		T sum = static_cast<T>(0);

		for (size_t i = 0; i < A.NumElements(); ++i) {
			sum += A[i] * B[i];
		}

		Memory::ArenaAllocator& allocator = A.GetAllocator();

		TensorCore::Tensor<T> C{ {1}, allocator };
		C[0] = sum;

		if (A.RequiresGrad() || B.RequiresGrad()) {
			C.SetRequiresGrad(true);
			C.SetGradFn(std::make_shared<AutoGrad::DotGradFn<T>>(A.GetImpl(), B.GetImpl()));
		}

		return C;
	}
}