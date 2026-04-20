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
		TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		for (size_t i = 0; i < A.NumElements(); ++i) {
			if (A[i] >= 0) {
				result[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-A[i]));
			}
			else {
				result[i] = std::exp(A[i]) / (static_cast<T>(1) + std::exp(A[i]));
			}
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::SigmoidGradFn<T>>(A.GetImpl(), result.GetImpl()));
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> Tanh(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		TensorCore::Tensor<T> result{ A.GetShape(), allocator };

		for (size_t i = 0; i < A.NumElements(); ++i) {
			result[i] = std::tanh(A[i]);
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::TanhGradFn<T>>(A.GetImpl(), result.GetImpl()));
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

		auto shape = A.GetShape(); // I'm intentionally making a copy here, because it's going to be done anyway
		TensorCore::Tensor<T> result{ shape, allocator };

		size_t axisSize = A.Dims()[axis];
		size_t outerSize = result.NumElements() / axisSize;

		for (size_t i = 0; i < outerSize; ++i) {
			auto baseIndex = shape.UnflattenIndex(i);
			baseIndex.insert(baseIndex.begin() + axis, 0);

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
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::AxisSoftmaxGradFn<T>>(A.GetImpl(), result.GetImpl(), axis));
		}

		return result;
	}
}