// scalar.inl
#include <type_traits>
#include <stdexcept>
#include <functional>

namespace MLCore::AutoGrad {
	template <typename T>
	class AddScalarGradFn;

	template <typename T>
	class SubScalarGradFn;

	template <typename T>
	class MulScalarGradFn;

	template <typename T>
	class DivScalarGradFn;
}

namespace MLCore::Operations {
	// Scalar Operations on RHS
	template <typename T>
	inline TensorCore::Tensor<T> AddScalar(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator) noexcept {
		TensorCore::Tensor<T> Output{ Input.GetShape(), allocator };

		const size_t size = Input.NumElements();

		for (size_t i = 0; i < size; ++i) {
			Output[i] = Input[i] + Scalar;
		}

		if (Input.RequiresGrad()) {
			Output.SetRequiresGrad(true);
			Output.SetGradFn(new AddScalarGradFn<T>(const_cast<TensorCore::Tensor<T>*>(&Input)));
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

		if (Input.RequiresGrad()) {
			Output.SetRequiresGrad(true);
			Output.SetGradFn(new SubScalarGradFn<T>(const_cast<TensorCore::Tensor<T>*>(&Input), false));
		}

		return Output;
	}

	template <typename T>
	inline TensorCore::Tensor<T> MultiplyScalar(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator) noexcept {
		TensorCore::Tensor<T> Output{ Input.GetShape(), allocator };

		const size_t size = Input.NumElements();

		for (size_t i = 0; i < size; ++i) {
			Output[i] = Input[i] * Scalar;
		}

		if (Input.RequiresGrad()) {
			Output.SetRequiresGrad(true);
			Output.SetGradFn(new MulScalarGradFn<T>(const_cast<TensorCore::Tensor<T>*>(&Input), Scalar));
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

		if (Input.RequiresGrad()) {
			Output.SetRequiresGrad(true);
			Output.SetGradFn(new MulScalarGradFn<T>(const_cast<TensorCore::Tensor<T>*>(&Input), Scalar, false));
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

		if (Input.RequiresGrad()) {
			Output.SetRequiresGrad(true);
			Output.SetGradFn(new SubScalarGradFn<T>(const_cast<TensorCore::Tensor<T>*>(&Input), true));
		}

		return Output;
	}

	template <typename T>
	inline TensorCore::Tensor<T> DivideScalarLeft(const T Scalar, const TensorCore::Tensor<T>& Input, Memory::ArenaAllocator& allocator) {
		if constexpr (!std::is_floating_point_v<T>) {
			for (size_t i = 0; i < Input.NumElements(); ++i) {
				if (Input[i] == 0) {
					throw std::runtime_error("ERROR: Divide by 0");
				}
			}
		}
			
		TensorCore::Tensor<T> Output{ Input.GetShape(), allocator };

		const size_t size = Input.NumElements();

		for (size_t i = 0; i < size; ++i) {
			Output[i] = Scalar / Input[i];
		}

		if (Input.RequiresGrad()) {
			Output.SetRequiresGrad(true);
			Output.SetGradFn(new MulScalarGradFn<T>(const_cast<TensorCore::Tensor<T>*>(&Input), Scalar, true));
		}

		return Output;
	}
}