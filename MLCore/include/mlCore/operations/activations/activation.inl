 /// activation.inl
#include <cmath>
#include <algorithm>
#include <mlCore/operations/reduction/reduction.h>
#include <mlCore/operations/broadcast/broadcast.h>
#include <mlCore/operations/elementwise/elementwise.h>
#include <mlCore/autograd/functions/activations/activationGradFn.h>

namespace MLCore::Operations {
	template <typename T>
	TensorCore::Tensor<T> ReLU(const TensorCore::Tensor<T>& A) {
		if (A.IsEmpty()) {
			throw std::runtime_error("ERROR: Input tensor cannot be empty");
		}

		Memory::ArenaAllocator& allocator = A.GetAllocator();
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
	TensorCore::Tensor<T> LeakyReLU(const TensorCore::Tensor<T>& A, T alpha) {
		if (A.IsEmpty()) {
			throw std::runtime_error("ERROR: Input tensor cannot be empty");
		}

		Memory::ArenaAllocator& allocator = A.GetAllocator();
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
	TensorCore::Tensor<T> Sigmoid(const TensorCore::Tensor<T>& A) {
		if (A.IsEmpty()) {
			throw std::runtime_error("ERROR: Input tensor cannot be empty");
		}

		const T safeBound = static_cast<T>(0.9) * static_cast<T>(std::log(std::numeric_limits<T>::max()));

		TensorCore::Tensor<T> clamped = Clamp(A, -safeBound, safeBound);
		TensorCore::Tensor<T> neg = Negate(clamped);
		TensorCore::Tensor<T> exp = Exp(neg);

		TensorCore::Tensor<T> sum = AddScalar(exp, static_cast<T>(1));
		TensorCore::Tensor<T> result = DivideScalar(sum, static_cast<T>(1), true);

		return result;
	}
	
	template <typename T>
	TensorCore::Tensor<T> Tanh(const TensorCore::Tensor<T>& A) {
		if (A.IsEmpty()) {
			throw std::runtime_error("ERROR: Input tensor cannot be empty");
		}

		const T safeBound = static_cast<T>(0.9) * static_cast<T>(std::log(std::numeric_limits<T>::max()));

		TensorCore::Tensor<T> clamped = Clamp(A, -safeBound, safeBound);

		TensorCore::Tensor<T> neg = Negate(clamped);

		TensorCore::Tensor<T> expPos = Exp(clamped); /// exp(x)
		TensorCore::Tensor<T> expNeg = Exp(neg); /// exp(-x)

		TensorCore::Tensor<T> diff = Subtract(expPos, expNeg);
		TensorCore::Tensor<T> sum = Add(expPos, expNeg);

		TensorCore::Tensor<T> result = Divide(diff, sum);

		return result;
	}
	
	template <typename T>
	TensorCore::Tensor<T> Softmax(const TensorCore::Tensor<T>& A) {
		if (A.IsEmpty()) {
			throw std::runtime_error("ERROR: Input tensor cannot be empty");
		}

		Memory::ArenaAllocator& allocator = A.GetAllocator();
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
	TensorCore::Tensor<T> AxisSoftmax(const TensorCore::Tensor<T>& A, size_t axis) {
		if (axis >= A.Rank()) {
			throw std::out_of_range("ERROR: AxisSoftmax: Axis out of bounds");
		}

		const std::vector<size_t>& dims = A.Dims();
		size_t rank = A.Rank();

		Memory::ArenaAllocator& allocator = A.GetAllocator();
		TensorCore::Tensor<T> result{ dims, allocator };
		result.Fill(static_cast<T>(0));

		/// Outer and inner size calculation
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

				T max = A[base];

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
	TensorCore::Tensor<T> AxisLogSoftmax(const TensorCore::Tensor<T>& A, size_t axis) {
		if (axis >= A.Rank()) {
			throw std::out_of_range("ERROR: AxisLogSoftmax: Axis out of bounds");
		}

		TensorCore::Tensor<T> axisMax = AxisMax(A, axis, true); /// For "numerical stability"
		TensorCore::Tensor<T> maxExpanded = ExpandToShape(axisMax, A.GetShape());

		TensorCore::Tensor<T> sub = Subtract(A, maxExpanded);

		TensorCore::Tensor<T> exp = Exp(sub);

		TensorCore::Tensor<T> sum = AxisSum(exp, axis, true);

		TensorCore::Tensor<T> log = Log(sum);

		TensorCore::Tensor<T> logExpanded = ExpandToShape(log, A.GetShape());
		TensorCore::Tensor<T> result = Subtract(sub, logExpanded);

		return result;
	}
}
