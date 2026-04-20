// scalar.inl
#include <concepts>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <mlCore/autograd/functions/scalar/scalarGradFn.h>

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
			Output.SetGradFn(std::make_shared<AutoGrad::AddScalarGradFn<T>>(Input.GetImpl()));
		}

		return Output;
	}

	template <typename T>
	inline TensorCore::Tensor<T> SubtractScalar(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator, bool scalarOnLeft) noexcept {
		TensorCore::Tensor<T> Output{ Input.GetShape(), allocator };

		const size_t size = Input.NumElements();

		for (size_t i = 0; i < size; ++i) {
			Output[i] = (scalarOnLeft) ? Scalar - Input[i] : Input[i] - Scalar;
		}

		if (Input.RequiresGrad()) {
			Output.SetRequiresGrad(true);
			Output.SetGradFn(std::make_shared<AutoGrad::SubScalarGradFn<T>>(Input.GetImpl(), scalarOnLeft));
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
			Output.SetGradFn(std::make_shared<AutoGrad::MulScalarGradFn<T>>(Input.GetImpl(), Scalar));
		}

		return Output;
	}

	template <typename T>
	inline TensorCore::Tensor<T> DivideScalar(const TensorCore::Tensor<T>& Input, const T Scalar, Memory::ArenaAllocator& allocator, bool scalarOnLeft) {
		// Also should create checks for if the tensor itself has any zeros
		if (Scalar == static_cast<T>(0) && !scalarOnLeft) {
			throw std::runtime_error("ERROR: Divide by 0");
		}

		TensorCore::Tensor<T> Output{ Input.GetShape(), allocator };

		const size_t size = Input.NumElements();

		for (size_t i = 0; i < size; ++i) {
			Output[i] = (scalarOnLeft) ? Scalar / Input[i] : Input[i] / Scalar;
		}

		if (Input.RequiresGrad()) {
			Output.SetRequiresGrad(true);
			Output.SetGradFn(std::make_shared<AutoGrad::DivScalarGradFn<T>>(Input.GetImpl(), Scalar, scalarOnLeft));
		}

		return Output;
	}
}