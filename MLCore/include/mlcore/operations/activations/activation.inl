// activation.inl
#include <cmath>
#include <algorithm>
#include <mlCore/autograd/functions/activations/activationGradFn.h>

namespace MLCore::Operations {
	template <typename T>
	TensorCore::Tensor<T> ReLU(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		for (size_t i = 0; i < A.NumElements(); ++i) {
			result[i] = std::max(static_cast<T>(0), A[i]);
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::ReLUGradFn<T>>(A.GetImpl()));
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> LeakyReLU(const TensorCore::Tensor<T>& A, T alpha, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		for (size_t i = 0; i < A.NumElements(); ++i) {
			result[i] = (A[i] > 0) ? A[i] : alpha * A[i];
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::LeakyReLUGradFn<T>>(A.GetImpl(), alpha));
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> Sigmoid(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		//TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		//// If the value x was >= 0, then result = 1/(1 + exp(-x))
		//for (size_t i = 0; i < A.NumElements(); ++i) {
		//	if (A[i] >= 0) {
		//		result[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-A[i]));
		//	}
		//	else {
		//		// Otherwise, result = exp(x)/(1 + exp(x))
		//		result[i] = std::exp(A[i]) / (static_cast<T>(1) + std::exp(A[i]));
		//	}
		//}

		TensorCore::Tensor<T> exp = Operations::Exp(A, allocator);
		TensorCore::Tensor<T> expPlusOne = Operations::AddScalar(exp, static_cast<T>(1), allocator);

		TensorCore::Tensor<T> result = Operations::Divide(exp, expPlusOne, allocator);

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			/*result.SetGradFn(std::make_shared<AutoGrad::SigmoidGradFn<T>>(A.GetImpl(), result.GetImpl()));*/
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> Tanh(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		/*TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		for (size_t i = 0; i < A.NumElements(); ++i) {
			result[i] = std::tanh(A[i]);
		}*/

		TensorCore::Tensor<T> neg = Operations::Negate(A, allocator);

		TensorCore::Tensor<T> expPos = Operations::Exp(A, allocator); // exp(x)
		TensorCore::Tensor<T> expNeg = Operations::Exp(neg, allocator); // exp(-x)

		TensorCore::Tensor<T> diff = Operations::Subtract(expPos, expNeg, allocator);
		TensorCore::Tensor<T> sum = Operations::Add(expPos, expNeg, allocator);

		TensorCore::Tensor<T> result = Operations::Divide(diff, sum, allocator);

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			/*result.SetGradFn(std::make_shared<AutoGrad::TanhGradFn<T>>(A.GetImpl(), result.GetImpl()));*/
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> Softmax(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
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

		// Outer and inner size calculation
		size_t outer = 1;
		for (size_t i = 0; i < axis; ++i) {
			outer *= dims[i];
		}

		size_t inner = 1;
		for (size_t i = 0; i < rank; ++i) {
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

		/*for (size_t i = 0; i < outerSize; ++i) {
			std::vector<size_t> baseIndex{ shape.Rank(), 0 };
			size_t temp = i;

			for (size_t j = shape.Rank(); j-- > 0;) {
				if (j == axis) {
					baseIndex[j] = 0;
				}

				baseIndex[j] = temp % shape.Dims()[j];
				temp /= shape.Dims()[j];
			}

			T max = A[shape.FlattenIndex(baseIndex)];

			for (size_t j = 0; j < axisSize; ++j) {
				baseIndex[axis] = j;
				max = std::max(max, A[shape.FlattenIndex(baseIndex)]);
			}

			T sumExp = static_cast<T>(0);

			for (size_t j = 0; j < axisSize; ++j) {
				baseIndex[axis] = j;
				T exp = std::exp(A[shape.FlattenIndex(baseIndex)] - max);
				result[shape.FlattenIndex(baseIndex)] = exp;
				sumExp += exp;
			}

			for (size_t j = 0; j < axisSize; ++j) {
				baseIndex[axis] = j;
				result[shape.FlattenIndex(baseIndex)] /= sumExp;
			}
		}*/

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::AxisSoftmaxGradFn<T>>(A.GetImpl(), result.GetImpl(), axis));
		}

		return result;
	}
}