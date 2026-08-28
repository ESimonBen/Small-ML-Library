 /// elementwiseGradFn.inl
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::AutoGrad {
	template <typename T>
	AddGradFn<T>::AddGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b)
		: GradFn<T>({ a, b })
	{}
	
	template <typename T>
	void AddGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (!this->inputs[0] || !this->inputs[1]) {
			throw std::runtime_error("ERROR: AddGradFn: Null input");
		}

		TensorCore::Tensor<T> a{ this->inputs[0] };
		TensorCore::Tensor<T> b{ this->inputs[1] };

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		if (a.RequiresGrad()) {
			auto gradA = Operations::ReduceSumToShape(gradientOut, a.GetShape());
			a.Backward(gradA);
		}

		if (b.RequiresGrad()) {
			auto gradB = Operations::ReduceSumToShape(gradientOut, b.GetShape());
			b.Backward(gradB);
		}
	}
	
	template <typename T>
	SubGradFn<T>::SubGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b)
		: GradFn<T>({ a, b })
	{}
	
	template <typename T>
	void SubGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (!this->inputs[0] || !this->inputs[1]) {
			throw std::runtime_error("ERROR: SubGradFn: Null input");
		}

		TensorCore::Tensor<T> a{ this->inputs[0] };
		TensorCore::Tensor<T> b{ this->inputs[1] };

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		if (a.RequiresGrad()) {
			auto gradA = Operations::ReduceSumToShape(gradientOut, a.GetShape());
			a.Backward(gradA);
		}

		if (b.RequiresGrad()) {
			auto gradB = Operations::ReduceSumToShape(Operations::Negate(gradientOut), b.GetShape());
			b.Backward(gradB);
		}
	}
	
	template <typename T>
	MulGradFn<T>::MulGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b)
		: GradFn<T>({ a, b })
	{}
	
	template <typename T>
	void MulGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (!this->inputs[0] || !this->inputs[1]) {
			throw std::runtime_error("ERROR: MulGradFn: Null input");
		}

		TensorCore::Tensor<T> a{ this->inputs[0] };
		TensorCore::Tensor<T> b{ this->inputs[1] };

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		

		if (a.RequiresGrad()) {
			auto detachedB = b.Detach();
			auto gradA = Operations::ReduceSumToShape(Operations::Multiply(gradientOut, detachedB), a.GetShape());
			a.Backward(gradA);
		}

		if (b.RequiresGrad()) {
			auto detachedA = a.Detach();
			auto gradB = Operations::ReduceSumToShape(Operations::Multiply(gradientOut, detachedA), b.GetShape());
			b.Backward(gradB);
		}
	}
	
	template <typename T>
	DivGradFn<T>::DivGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, std::shared_ptr<typename GradFn<T>::Impl> b)
		: GradFn<T>({ a, b })
	{}
	
	template <typename T>
	void DivGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (!this->inputs[0] || !this->inputs[1]) {
			throw std::runtime_error("ERROR: DivGradFn: Null input");
		}

		TensorCore::Tensor<T> a{ this->inputs[0] };
		TensorCore::Tensor<T> b{ this->inputs[1] };

		
		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		if (a.RequiresGrad()) {
			auto detachedB = b.Detach();
			auto gradA = Operations::ReduceSumToShape(Operations::Divide(gradientOut, detachedB), a.GetShape());
			a.Backward(gradA);
		}

		if (b.RequiresGrad()) {
			auto detachedA = a.Detach();
			auto detachedB = b.Detach();

			auto negGradOutput = Operations::Negate(gradientOut);
			auto bSquared = Operations::Square(detachedB);
			
			auto gradB = Operations::ReduceSumToShape(Operations::Multiply(negGradOutput, Operations::Divide(detachedA, bSquared)), b.GetShape());
			b.Backward(gradB);
		}
	}
	
	template <typename T>
	PowerGradFn<T>::PowerGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T exponent)
		: GradFn<T>(a), m_Exponent(exponent)
	{}
	
	template <typename T>
	void PowerGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (!this->inputs[0]) {
			throw std::runtime_error("ERROR: PowerGradFn: Null input");
		}

		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		auto base = input.Detach();

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		auto coeff = Operations::MultiplyScalar(gradientOut, m_Exponent);
		auto expMinus1 = Operations::Power(base, m_Exponent - static_cast<T>(1));

		TensorCore::Tensor<T> gradInput = Operations::Multiply(coeff, expMinus1);

		input.Backward(gradInput);
	}
	
	template <typename T>
	AbsGradFn<T>::AbsGradFn(std::shared_ptr<typename GradFn<T>::Impl> input)
		: GradFn<T>(input)
	{}
	
	template <typename T>
	void AbsGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (!this->inputs[0]) {
			throw std::runtime_error("ERROR: AbsGradFn: Null input");
		}

		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradInput{ input.GetShape(), input.GetAllocator() };

		size_t size = gradInput.NumElements();

		for (size_t i = 0; i < size; ++i) {
			T inp = input[i];

			T sign = (inp < static_cast<T>(0)) ? static_cast<T>(-1) : ((inp > static_cast<T>(0)) ? static_cast<T>(1) : static_cast<T>(0));
			gradInput[i] = gradOutput[i] * sign;
		}

		input.Backward(gradInput);
	}
	
	template <typename T>
	ClampGradFn<T>::ClampGradFn(std::shared_ptr<typename GradFn<T>::Impl> input, T min, T max)
		: GradFn<T>(input), m_Min(min), m_Max(max)
	{}
	
	template <typename T>
	void ClampGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (!this->inputs[0]) {
			throw std::runtime_error("ERROR: ClampGradFn: Null input");
		}

		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradInput{ input.GetShape(), input.GetAllocator() };

		size_t size = gradInput.NumElements();

		for (size_t i = 0; i < size; ++i) {
			T inp = input[i];

			if (inp > m_Min && inp < m_Max) {
				gradInput[i] = gradOutput[i];
			}
			else {
				gradInput[i] = static_cast<T>(0);
			}
		}

		input.Backward(gradInput);
	}
	
	template <typename T>
	LogGradFn<T>::LogGradFn(std::shared_ptr<typename GradFn<T>::Impl> input)
		: GradFn<T>(input)
	{}
	
	template <typename T>
	void LogGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (!this->inputs[0]) {
			throw std::runtime_error("ERROR: LogGradFn: Null input");
		}

		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> inp = input.Detach();
		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		TensorCore::Tensor<T> reciprocal = Operations::DivideScalar(inp, static_cast<T>(1), true);
		TensorCore::Tensor<T> gradInput = Operations::Multiply(gradientOut, reciprocal);

		input.Backward(gradInput);
	}
	
	template <typename T>
	ExpGradFn<T>::ExpGradFn(std::shared_ptr<typename GradFn<T>::Impl> input)
		: GradFn<T>(input)
	{}
	
	template <typename T>
	void ExpGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (!this->inputs[0]) {
			throw std::runtime_error("ERROR: LogGradFn: Null input");
		}

		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> inp = input.Detach();
		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		TensorCore::Tensor<T> exponential = Operations::Exp(inp);
		TensorCore::Tensor<T> gradInput = Operations::Multiply(exponential, gradientOut);

		input.Backward(gradInput);
	}
}