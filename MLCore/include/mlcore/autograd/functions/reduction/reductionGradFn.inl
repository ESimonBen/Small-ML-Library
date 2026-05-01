// reductionGradFn.inl
#include <stdexcept>
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/broadcast/broadcast.h>

namespace MLCore::AutoGrad {
	template <typename T>
	SumGradFn<T>::SumGradFn(std::shared_ptr<typename GradFn<T>::Impl> a)
		: GradFn<T>(a), inputShape(a->shape)
	{}

	template <typename T>
	void SumGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: SumGradFn: Expected scalar tensor");
		}

		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor gradientOut = gradOutput.Detach();

		auto gradInput = ExpandToShape(gradientOut, inputShape);

		input.Backward(gradInput);
	}

	template <typename T>
	MaxGradFn<T>::MaxGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T maxValue)
		: GradFn<T>(a), inputShape(a->shape), maxValue(maxValue)
	{}

	template <typename T>
	void MaxGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: MaxGradFn: Expected a 1D tensor");
		}

		TensorCore::Tensor<T> input{this->inputs[0]};

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> gradInput{ inputShape, allocator };

		T gradScalar = gradOutput[0];

		size_t size = gradInput.NumElements();
		size_t count = 0;

		// May change implementation to split the gradient over all the max elements
		// in the tensor rather than setting all of the max elements to the max gradient (for stability's sake)
		for (size_t i = 0; i < size; ++i) {
			if (input[i] == maxValue) {
				count++;
			}
		}

		for (size_t i = 0; i < size; ++i) {
			gradInput[i] = (input[i] == maxValue) ? gradScalar / count : 0;
		}

		input.Backward(gradInput);
	}

	template <typename T>
	MinGradFn<T>::MinGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, T minValue)
		: GradFn<T>(a), inputShape(a->shape), minValue(minValue)
	{}

	template <typename T>
	void MinGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) {
		if (gradOutput.NumElements() != 1) {
			throw std::runtime_error("ERROR: MinGradFn: Expected a 1D tensor");
		}

		TensorCore::Tensor<T> input{this->inputs[0]};

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();
		TensorCore::Tensor<T> gradInput{ inputShape, allocator };

		T gradScalar = gradOutput[0];

		size_t size = gradInput.NumElements();
		size_t count = 0;

		// May change implementation to split the gradient over all the max elements in the tensor 
		// rather than setting all of the max elements to the max gradient (for stability's sake)
		for (size_t i = 0; i < size; ++i) {
			if (input[i] == minValue) {
				count++;
			}
		}

		for (size_t i = 0; i < size; ++i) {
			gradInput[i] = (input[i] == minValue) ? gradScalar / count : 0;
		}

		input.Backward(gradInput);
	}

	template <typename T>
	AxisSumGradFn<T>::AxisSumGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis, bool keepDims)
		: GradFn<T>(a), inputShape(a->shape), m_Axis(axis), m_KeepDims(keepDims) {
		assert(axis < inputShape.Rank());
	}

	template <typename T>
	void AxisSumGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> input{this->inputs[0]};

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		if (!m_KeepDims) {
			gradientOut = Operations::Unsqueeze(gradientOut, m_Axis, allocator);
		}

		TensorCore::Tensor<T> gradInput = ExpandToShape(gradientOut, inputShape);

		input.Backward(gradInput);
	}

	template <typename T>
	AxisMaxGradFn<T>::AxisMaxGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis, bool keepDims)
		: GradFn<T>(a), m_Axis(axis), m_KeepDims(keepDims)
	{}

	template <typename T>
	void AxisMaxGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> inp = input.Detach();
		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		if (!m_KeepDims) {
			gradientOut = Operations::Unsqueeze(gradientOut, m_Axis, allocator);
		}

		TensorCore::Tensor<T> axisMax = Operations::AxisMax(inp, m_Axis, allocator, true); // Recalculating the max we got (maybe save the max ahead of time rather than recalculating)
		TensorCore::Tensor<T> maxExpanded = ExpandToShape(axisMax, inp.GetShape());

		TensorCore::Tensor<T> gradExpanded = ExpandToShape(gradientOut, input.GetShape());

		TensorCore::Tensor<T> mask = Operations::Equal(input, maxExpanded, allocator);

		TensorCore::Tensor<T> count = Operations::AxisSum(mask, m_Axis, allocator, true);
		TensorCore::Tensor<T> countExpanded = ExpandToShape(count, inp.GetShape());

		TensorCore::Tensor<T> div = Operations::Divide(gradExpanded, countExpanded, allocator);
		TensorCore::Tensor<T> gradInput = Operations::Multiply(mask, div, allocator);

		input.Backward(gradInput);
	}

	template <typename T>
	AxisMinGradFn<T>::AxisMinGradFn(std::shared_ptr<typename GradFn<T>::Impl> a, size_t axis, bool keepDims)
		: GradFn<T>(a), m_Axis(axis), m_KeepDims(keepDims)
	{}

	template <typename T>
	void AxisMinGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> input{ this->inputs[0] };

		if (!input.RequiresGrad()) {
			return;
		}

		TensorCore::Tensor<T> inp = input.Detach();
		TensorCore::Tensor<T> gradientOut = gradOutput.Detach();

		if (!m_KeepDims) {
			gradientOut = Operations::Unsqueeze(gradientOut, m_Axis, allocator);
		}

		TensorCore::Tensor<T> axisMin = Operations::AxisMin(inp, m_Axis, allocator, true); // Recalculating the max we got (maybe save the max ahead of time rather than recalculating)
		TensorCore::Tensor<T> minExpanded = ExpandToShape(axisMin, inp.GetShape());

		TensorCore::Tensor<T> gradExpanded = ExpandToShape(gradientOut, input.GetShape());

		TensorCore::Tensor<T> mask = Operations::Equal(input, minExpanded, allocator);

		TensorCore::Tensor<T> count = Operations::AxisSum(mask, m_Axis, allocator, true);
		TensorCore::Tensor<T> countExpanded = ExpandToShape(count, inp.GetShape());

		TensorCore::Tensor<T> div = Operations::Divide(gradExpanded, countExpanded, allocator);
		TensorCore::Tensor<T> gradInput = Operations::Multiply(mask, div, allocator);

		input.Backward(gradInput);
	}
}