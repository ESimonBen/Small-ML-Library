// loss.inl
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <mlCore/autograd/functions/loss/lossGradFn.h>

namespace MLCore::Operations {
	template <typename T>
	inline TensorCore::Tensor<T> MeanSquaredError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator) {
		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: MeanSquaredError: Tensor size mismatch");
		}

		T sum = T{};

		size_t count = predictions.NumElements();
		for (size_t i = 0; i < count; ++i) {
			T diff = targets[i] - predictions[i];
			sum += diff * diff;
		}

		TensorCore::Tensor<T> result{ {1}, allocator };
		result[0] = sum / predictions.NumElements();

		if (predictions.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::MSEGradFn<T>>(predictions.GetImpl(), targets.GetImpl()));
		}

		return result;
	}

	template <typename T>
	inline TensorCore::Tensor<T> MeanAbsoluteError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator) {
		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: MeanAbsoluteError: Tensor size mismatch");
		}

		T sum = T{};

		size_t count = predictions.NumElements();
		for (size_t i = 0; i < count; ++i) {
			sum += std::abs(targets[i] - predictions[i]);
		}

		TensorCore::Tensor<T> result{ {1}, allocator };
		result[0] = sum / predictions.NumElements();

		if (predictions.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::MAEGradFn<T>>(predictions.GetImpl(), targets.GetImpl()));
		}

		return result;
	}

	template <typename T>
	inline TensorCore::Tensor<T> BinaryCrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator) {
		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: BinaryCrossEntropy: Tensor size mismatch");
		}

		T sum = T{};
		const T epsilon = static_cast<T>(1e-7);

		size_t count = predictions.NumElements();
		for (size_t i = 0; i < count; ++i) {
			T p = std::clamp(predictions[i], epsilon, 1 - epsilon);
			sum += -(targets[i] * std::log(p) + (1 - targets[i]) * std::log(1 - p));
		}

		TensorCore::Tensor<T> result{ {1}, allocator };
		result[0] = sum /*/ predictions.NumElements()*/;

		if (predictions.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::BCEGradFn<T>>(predictions.GetImpl(), targets.GetImpl()));
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> BinaryCrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator) {
		if (logits.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: BinaryCrossEntropyWithLogits: Tensor shape mismatch");
		}

		T sum = static_cast<T>(0);
		size_t size = logits.NumElements();

		for (size_t i = 0; i < size; ++i) {
			T logit = logits[i];
			T target = targets[i];

			T max = std::max(logit, static_cast<T>(0));
			T logTerm = std::log(std::exp(-std::abs(logit)) + static_cast<T>(1));

			sum += max - logit * target + logTerm;
		}

		TensorCore::Tensor<T> result{ {1}, allocator };

		result[0] = sum / size;

		if (logits.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::BCEWithLogitsGradFn<T>>(logits.GetImpl(), targets.GetImpl()));
		}

		return result;
	}

	// This assumes that the result of Softmax is being passed in
	// (Softmax returns the probabilities of each element of the prediction)
	template <typename T>
	inline TensorCore::Tensor<T> CrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator) {
		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: CrossEntropy: Tensor shape mismatch");
		}

		T sum = T{};
		const T epsilon = static_cast<T>(1e-7);

		size_t count = predictions.NumElements();
		for (size_t i = 0; i < count; ++i) {
			T p = std::clamp(predictions[i], epsilon, 1 - epsilon);
			sum += -targets[i] * std::log(p); // std::log should have been named std::ln, but WHAT DO I KNOW
		}

		TensorCore::Tensor<T> result{ {1}, allocator };
		result[0] = sum / predictions.NumElements();

		if (predictions.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::CEGradFn<T>>(predictions.GetImpl(), targets.GetImpl()));
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> CrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, size_t axis, Memory::ArenaAllocator& allocator) {
		if (logits.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: CrossEntropyWithLogits: Tensor shape mismatch");
		}

		if (axis >= logits.Rank()) {
			throw std::out_of_range("ERROR: AxisSum: Axis out of bounds");
		}

		auto shape = logits.GetShape();
		size_t axisSize = logits.Dims()[axis];
		size_t outerSize = logits.NumElements() / axisSize;

		T totalLoss = static_cast<T>(0);

		for (size_t i = 0; i < outerSize; ++i) {
			std::vector<size_t> baseIndex{ shape.Rank(), 0 };
			size_t temp = i;
			
			for (size_t j = shape.Rank(); j-- > 0;) {
				if (j == axis) {
					baseIndex[j] = 0;
				}

				baseIndex[j] = temp % shape.Dims()[j];
				temp /= shape.Dims()[j];
			}

			T max = -std::numeric_limits<T>::infinity();

			for (size_t j = 0; j < axisSize; ++j) {
				baseIndex[axis] = j;
				max = std::max(max, logits[shape.FlattenIndex(baseIndex)]);
			}

			T sumExp = static_cast<T>(0);

			for (size_t j = 0; j < axisSize; ++j) {
				baseIndex[axis] = j;
				sumExp += std::exp(logits[shape.FlattenIndex(baseIndex)] - max);
			}

			T logSumExp = std::log(sumExp) + max;

			for (size_t j = 0; j < axisSize; ++j) {
				baseIndex[axis] = j;
				size_t idx = shape.FlattenIndex(baseIndex);
				T target = targets[idx];
				totalLoss += -target * (logits[idx] - logSumExp);
			}
		}

		TensorCore::Tensor<T> result{ {1}, allocator };
		result[0] = totalLoss / outerSize;

		if (logits.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::CEWithLogitsGradFn<T>>(logits.GetImpl(), targets.GetImpl(), axis));
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> CrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator) {
		return CrossEntropyWithLogits(logits, targets, logits.Rank() - 1, allocator);
	}
}