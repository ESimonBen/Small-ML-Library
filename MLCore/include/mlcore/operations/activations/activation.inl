 /// activation.inl
#include <cmath>
#include <algorithm>
#include <mlCore/operations/reduction/reduction.h>
#include <mlCore/operations/broadcast/broadcast.h>
#include <mlCore/operations/elementwise/elementwise.h>
#include <mlCore/autograd/functions/activations/activationGradFn.h>

namespace MLCore::Operations {
	template <typename T>
	TensorCore::Tensor<T> ReLU(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		if (A.Dims().empty()) {
			throw std::runtime_error("ERROR: Input tensor cannot be empty");
		}

		TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		size_t size = A.NumElements();

		for (size_t i = 0; i < size; ++i) {
			T testVal = A[i];
			result[i] = (testVal > static_cast<T>(0)) ? testVal : static_cast<T>(0);
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::ReLUGradFn<T>>(A.GetImpl()));
		}

		return result;
	}
	
	template <typename T>
	TensorCore::Tensor<T> LeakyReLU(const TensorCore::Tensor<T>& A, T alpha, Memory::ArenaAllocator& allocator) {
		if (A.Dims().empty()) {
			throw std::runtime_error("ERROR: Input tensor cannot be empty");
		}

		TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		size_t size = A.NumElements();

		for (size_t i = 0; i < size; ++i) {
			T testVal = A[i];
			result[i] = (testVal > static_cast<T>(0)) ? testVal : alpha * testVal;
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::LeakyReLUGradFn<T>>(A.GetImpl(), alpha));
		}

		return result;
	}
	
	template <typename T>
	TensorCore::Tensor<T> Sigmoid(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		if (A.Dims().empty()) {
			throw std::runtime_error("ERROR: Input tensor cannot be empty");
		}

		TensorCore::Tensor<T> neg = Negate(A, allocator);
		TensorCore::Tensor<T> exp = Exp(neg, allocator);

		TensorCore::Tensor<T> sum = AddScalar(exp, static_cast<T>(1), allocator);
		TensorCore::Tensor<T> result = DivideScalar(sum, static_cast<T>(1), allocator, true);

		return result;
	}
	
	template <typename T>
	TensorCore::Tensor<T> Tanh(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		if (A.Dims().empty()) {
			throw std::runtime_error("ERROR: Input tensor cannot be empty");
		}

		TensorCore::Tensor<T> neg = Negate(A, allocator);

		TensorCore::Tensor<T> expPos = Exp(A, allocator); /// exp(x)
		TensorCore::Tensor<T> expNeg = Exp(neg, allocator); /// exp(-x)

		TensorCore::Tensor<T> diff = Subtract(expPos, expNeg, allocator);
		TensorCore::Tensor<T> sum = Add(expPos, expNeg, allocator);

		TensorCore::Tensor<T> result = Divide(diff, sum, allocator);

		return result;
	}
	
	template <typename T>
	TensorCore::Tensor<T> Softmax(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		if (A.Dims().empty()) {
			throw std::runtime_error("ERROR: Input tensor cannot be empty");
		}

		TensorCore::Tensor<T> result{ A.GetShape(), allocator };
		size_t size = A.NumElements();
		T maxValue = A[0];

		for (size_t i = 0; i < size; ++i) {
			if (A[i] > maxValue) {
				maxValue = A[i];
			}
		}

		T sumExp = static_cast<T>(0);

		for (size_t i = 0; i < size; ++i) {
			result[i] = std::exp(A[i] - maxValue);
			sumExp += result[i];
		}

		for (size_t i = 0; i < size; ++i) {
			result[i] /= sumExp;
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::SoftmaxGradFn<T>>(A.GetImpl(), result.GetImpl()));
		}

		return result;
	}
	
	template <typename T>
	TensorCore::Tensor<T> AxisSoftmax(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator) {
		if (axis >= A.Rank()) {
			throw std::out_of_range("ERROR: AxisSoftmax: Axis out of bounds");
		}

		const std::vector<size_t>& dims = A.Dims();
		size_t rank = A.Rank();

		TensorCore::Tensor<T> result{ dims, allocator };
		result.Fill(static_cast<T>(0));

		// Outer and inner size calculation
		size_t outer = 1;
		for (size_t i = 0; i < axis; ++i) {
			outer *= dims[i];
		}

		size_t inner = 1;
		for (size_t i = axis + 1; i < rank; ++i) {
			inner *= dims[i];
		}

		size_t axisSize = dims[axis];

		for (size_t o = 0; o < outer; ++o) {
			for (size_t i = 0; i < inner; ++i) {
				size_t base = o * axisSize * inner + i;

				T max = -std::numeric_limits<T>::infinity();

				for (size_t j = 0; j < axisSize; ++j) {
					T testVal = A[base + j * inner];
					max = (max > testVal) ? max : testVal;
				}

				T sumExp = static_cast<T>(0);
				for (size_t j = 0; j < axisSize; ++j) {
					T exp = std::exp(A[base + j * inner] - max);
					result[base + j * inner] = exp;
					sumExp += exp;
				}

				for (size_t j = 0; j < axisSize; ++j) {
					result[base + j * inner] /= sumExp;
				}
			}
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::AxisSoftmaxGradFn<T>>(A.GetImpl(), result.GetImpl(), axis));
		}

		return result;
	}
	
	template <typename T>
	TensorCore::Tensor<T> AxisLogSoftmax(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator) {
		if (axis >= A.Rank()) {
			throw std::out_of_range("ERROR: AxisLogSoftmax: Axis out of bounds");
		}

		TensorCore::Tensor<T> axisMax = AxisMax(A, axis, allocator, true); /// For "numerical stability"
		TensorCore::Tensor<T> maxExpanded = ExpandToShape(axisMax, A.GetShape(), allocator);

		TensorCore::Tensor<T> sub = Subtract(A, maxExpanded, allocator);

		TensorCore::Tensor<T> exp = Exp(sub, allocator);

		TensorCore::Tensor<T> sum = AxisSum(exp, axis, allocator, true);

		TensorCore::Tensor<T> log = Log(sum, allocator);

		TensorCore::Tensor<T> logExpanded = ExpandToShape(log, A.GetShape(), allocator);
		TensorCore::Tensor<T> result = Subtract(sub, logExpanded, allocator);

		return result;
	}
}
