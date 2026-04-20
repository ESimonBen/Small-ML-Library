// elementwiseGradFn.inl
#include <mlCore/autograd/gradientUtils.h>
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::AutoGrad {
	template <typename T>
	AddGradFn<T>::AddGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b)
		: GradFn<T>({ a, b })
	{}

	template <typename T>
	void AddGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		#ifdef ML_CORE_DEBUG
			if (!this->inputs[0] || !this->inputs[1]) {
				throw std::runtime_error("ERROR: AddGradFn: Null input");
			}
		#endif

		TensorCore::Tensor<T> a{ this->inputs[0] };
		TensorCore::Tensor<T> b{ this->inputs[1] };

		if (a.RequiresGrad()) {
			auto gradA = ReduceSumToShape(gradOutput, a.GetShape());
			a.Backward(gradA);
		}

		if (b.RequiresGrad()) {
			auto gradB = ReduceSumToShape(gradOutput, b.GetShape());
			b.Backward(gradB);
		}
	}

	template <typename T>
	SubGradFn<T>::SubGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b)
		: GradFn<T>({ a, b })
	{}

	template <typename T>
	void SubGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		#ifdef ML_CORE_DEBUG
			if (!this->inputs[0] || !this->inputs[1]) {
				throw std::runtime_error("ERROR: SubGradFn: Null input");
			}
		#endif

		TensorCore::Tensor<T> a{ this->inputs[0] };
		TensorCore::Tensor<T> b{ this->inputs[1] };

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		if (a.RequiresGrad()) {
			auto gradA = ReduceSumToShape(gradientOut, a.GetShape());
			a.Backward(gradA);
		}

		if (b.RequiresGrad()) {
			auto gradB = ReduceSumToShape(Operations::Negate(gradientOut, gradientOut.GetAllocator()), b.GetShape());
			b.Backward(gradB);
		}
	}

	template <typename T>
	MulGradFn<T>::MulGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b)
		: GradFn<T>({ a, b })
	{}

	template <typename T>
	void MulGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		#ifdef ML_CORE_DEBUG
			if (!this->inputs[0] || !this->inputs[1]) {
				throw std::runtime_error("ERROR: MulGradFn: Null input");
			}
		#endif

		TensorCore::Tensor<T> a{ this->inputs[0] };
		TensorCore::Tensor<T> b{ this->inputs[1] };

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		if (a.RequiresGrad()) {
			auto detachedB = b.Detach();
			auto gradA = ReduceSumToShape(Operations::Multiply(gradientOut, detachedB, allocator), a.GetShape());
			a.Backward(gradA);
		}

		if (b.RequiresGrad()) {
			auto detachedA = a.Detach();
			auto gradB = ReduceSumToShape(Operations::Multiply(gradientOut, detachedA, allocator), b.GetShape());
			b.Backward(gradB);
		}
	}

	template <typename T>
	DivGradFn<T>::DivGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b)
		: GradFn<T>({ a, b })
	{}

	template <typename T>
	void DivGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		#ifdef ML_CORE_DEBUG
			if (!this->inputs[0] || !this->inputs[1]) {
				throw std::runtime_error("ERROR: DivGradFn: Null input");
			}
		#endif

		TensorCore::Tensor<T> a{ this->inputs[0] };
		TensorCore::Tensor<T> b{ this->inputs[1] };

		
		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		if (a.RequiresGrad()) {
			auto detachedB = b.Detach();
			auto gradA = ReduceSumToShape(Operations::Divide(gradientOut, detachedB, allocator), a.GetShape());
			a.Backward(gradA);
		}

		if (b.RequiresGrad()) {
			auto detachedA = a.Detach();
			auto detachedB = b.Detach();

			auto negGradOutput = Operations::Negate(gradientOut, allocator);
			auto bSquared = Operations::Square(detachedB, allocator);
			
			auto gradB = ReduceSumToShape(Operations::Multiply(negGradOutput, Operations::Divide(detachedA, bSquared, allocator), allocator), b.GetShape());
			b.Backward(gradB);
		}
	}

	template <typename T>
	PowerGradFn<T>::PowerGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T exponent)
		: GradFn<T>(a), exponent(exponent)
	{}

	template <typename T>
	void PowerGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		#ifdef ML_CORE_DEBUG
			if (!this->inputs[0]) {
				throw std::runtime_error("ERROR: PowerGradFn: Null input");
			}
		#endif

		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		auto base = input.Detach();

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		auto& allocator = gradientOut.GetAllocator();

		auto coeff = Operations::MultiplyScalar(gradientOut, exponent, allocator);
		auto expMinus1 = Operations::Power(base, exponent - static_cast<T>(1), allocator);

		TensorCore::Tensor<T> gradInput = Operations::Multiply(coeff, expMinus1, allocator);

		input.Backward(gradInput);
	}
}