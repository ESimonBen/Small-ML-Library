// scalar.inl
#include <type_traits>
#include <stdexcept>
#include <functional>

namespace MLCore::Operations {
	// Scalar Operations on RHS
	template <typename T>
	inline TensorCore::Tensor<T> AddScalarRight(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator) noexcept {
		TensorCore::Tensor<T> Output{ Input.GetShape(), allocator };

		const size_t size = Input.NumElements();

		for (size_t i = 0; i < size; ++i) {
			Output[i] = Input[i] + Scalar;
		}

		return Output;
	}

	template <typename T>
	inline TensorCore::Tensor<T> SubtractScalarRight(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator) noexcept {
		TensorCore::Tensor<T> Output{ Input.GetShape(), allocator };

		const size_t size = Input.NumElements();

		for (size_t i = 0; i < size; ++i) {
			Output[i] = Input[i] - Scalar;
		}

		return Output;
	}

	template <typename T>
	inline TensorCore::Tensor<T> MultiplyScalarRight(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator) noexcept {
		TensorCore::Tensor<T> Output{ Input.GetShape(), allocator };

		const size_t size = Input.NumElements();

		for (size_t i = 0; i < size; ++i) {
			Output[i] = Input[i] * Scalar;
		}

		return Output;
	}

	template <typename T>
	inline TensorCore::Tensor<T> DivideScalarRight(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator) {
		if constexpr (!std::is_floating_point_v<T>) {
			if (Scalar == 0) {
				throw std::runtime_error("ERROR: Divide by 0");
			}
		}

		TensorCore::Tensor<T> Output{ Input.GetShape(), allocator };

		const size_t size = Input.NumElements();

		for (size_t i = 0; i < size; ++i) {
			Output[i] = Input[i] / Scalar;
		}

		return Output;
	}

	// Scalar Operations on LHS
	template <typename T>
	inline TensorCore::Tensor<T> AddScalarLeft(const T Scalar, const TensorCore::Tensor<T>& Input, Memory::ArenaAllocator& allocator) noexcept {
		TensorCore::Tensor<T> Output{ Input.GetShape(), allocator };

		const size_t size = Input.NumElements();

		for (size_t i = 0; i < size; ++i) {
			Output[i] = Scalar + Input[i];
		}

		return Output;
	}

	template <typename T>
	inline TensorCore::Tensor<T> SubtractScalarLeft(const T Scalar, const TensorCore::Tensor<T>& Input, Memory::ArenaAllocator& allocator) noexcept {
		TensorCore::Tensor<T> Output{ Input.GetShape(), allocator };

		const size_t size = Input.NumElements();

		for (size_t i = 0; i < size; ++i) {
			Output[i] = Scalar - Input[i];
		}

		return Output;
	}

	template <typename T>
	inline TensorCore::Tensor<T> MultiplyScalarLeft(const T Scalar, const TensorCore::Tensor<T>& Input, Memory::ArenaAllocator& allocator) noexcept {
		TensorCore::Tensor<T> Output{ Input.GetShape(), allocator };

		const size_t size = Input.NumElements();

		for (size_t i = 0; i < size; ++i) {
			Output[i] = Scalar * Input[i];
		}

		return Output;
	}

	template <typename T>
	inline TensorCore::Tensor<T> DivideScalarLeft(const T Scalar, const TensorCore::Tensor<T>& Input, Memory::ArenaAllocator& allocator) {
		if constexpr (!std::is_floating_point_v<T>) {
			if (Scalar == 0) {
				throw std::runtime_error("ERROR: Divide by 0");
			}
		}
			
		TensorCore::Tensor<T> Output{ Input.GetShape(), allocator };

		const size_t size = Input.NumElements();

		for (size_t i = 0; i < size; ++i) {
			Output[i] = Scalar / Input[i];
		}

		return Output;
	}
}