// elementwiseGradFn.inl

namespace MLCore::AutoGrad {
	template <typename T>
	AddGradFn<T>::AddGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
		: GradFn<T>({ a, b })
	{}

	template <typename T>
	void AddGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* a = this->inputs[0];
		auto* b = this->inputs[1];


		if (a->RequiresGrad()) {
			auto gradA = ReduceSumToShape(gradOutput, a->GetShape());
			a->Backward(gradA);
		}

		if (b->RequiresGrad()) {
			auto gradB = ReduceSumToShape(gradOutput, b->GetShape());
			b->Backward(gradB);
		}
	}

	template <typename T>
	SubGradFn<T>::SubGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
		: GradFn<T>({ a, b })
	{}

	template <typename T>
	void SubGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* a = this->inputs[0];
		auto* b = this->inputs[1];

		if (a->RequiresGrad()) {
			auto gradA = ReduceSumToShape(gradOutput, a->GetShape());
			a->Backward(gradA);
		}

		if (b->RequiresGrad()) {
			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());
			auto gradB = ReduceSumToShape(Operations::Negate(gradOutput, allocator), b->GetShape());
			b->Backward(gradB);
		}
	}

	template <typename T>
	MulGradFn<T>::MulGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
		: GradFn<T>({ a, b })
	{}

	template <typename T>
	void MulGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* a = this->inputs[0];
		auto* b = this->inputs[1];

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		if (a->RequiresGrad()) {
			auto detachedB = b->Detach();

			auto gradA = ReduceSumToShape(Operations::Multiply(gradOutput, detachedB, allocator), a->GetShape());
			a->Backward(gradA);

		}

		if (b->RequiresGrad()) {
			auto detachedA = a->Detach();

			auto gradB = ReduceSumToShape(Operations::Multiply(gradOutput, detachedA, allocator), b->GetShape());
			b->Backward(gradB);

		}
	}

	template <typename T>
	DivGradFn<T>::DivGradFn(TensorCore::Tensor<T>* a, TensorCore::Tensor<T>* b)
		: GradFn<T>({ a, b })
	{}

	template <typename T>
	void DivGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* a = this->inputs[0];
		auto* b = this->inputs[1];

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		if (a->RequiresGrad()) {
			auto detachedB = b->Detach();
			auto gradA = ReduceSumToShape(Operations::Divide(gradOutput, detachedB, allocator), a->GetShape());
			a->Backward(gradA);

		}

		if (b->RequiresGrad()) {
			auto detachedA = a->Detach();
			auto detachedB = b->Detach();

			auto negGradOutput = Operations::Negate(gradOutput, allocator);
			auto bSquared = Operations::Square(detachedB, allocator);

			auto gradB = ReduceSumToShape(Operations::Multiply(negGradOutput, Operations::Divide(detachedA, bSquared, allocator), allocator), b->GetShape());
			b->Backward(gradB);

		}
	}

	template <typename T>
	PowerGradFn<T>::PowerGradFn(TensorCore::Tensor<T>* a, T exponent)
		: GradFn<T>(a), exponent(exponent)
	{}

	template <typename T>
	void PowerGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		auto* input = this->inputs[0];

		if (!input->RequiresGrad()) {
			return;
		}

		auto base = input->Detach();

		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradOutput.GetAllocator());

		auto coeff = Operations::MultiplyScalar(gradOutput, exponent, allocator);
		auto expMinus1 = Operations::Power(base, exponent - static_cast<T>(1), allocator);

		TensorCore::Tensor<T> gradInput = Operations::Multiply(coeff, expMinus1, allocator);

		input->Backward(gradInput);
	}
}