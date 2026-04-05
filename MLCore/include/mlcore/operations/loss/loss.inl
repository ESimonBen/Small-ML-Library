// loss.inl
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace MLCore::Operations {
	template <typename T>
	inline TensorCore::Tensor<T> MeanSquaredError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator) {
		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: MeanSquaredError: Tensor size mismatch");
		}

		T sum = T{};

		size_t count = predictions.NumElements();
		for (size_t i = 0; i < count; ++i) {
			T diff = predictions[i] - targets[i];
			sum += diff * diff;
		}

		TensorCore::Tensor<T> result{ {1}, allocator };
		result[0] = sum / predictions.NumElements();

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
			sum += std::abs(predictions[i] - targets[i]);
		}

		TensorCore::Tensor<T> result{ {1}, allocator };
		result[0] = sum / predictions.NumElements();

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
		result[0] = sum / predictions.NumElements();

		return result;
	}

	template <typename T>
	inline TensorCore::Tensor<T> CrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Memory::ArenaAllocator& allocator) {
		if (predictions.GetShape() != targets.GetShape()) {
			throw std::runtime_error("ERROR: CrossEntropy: Tensor size mismatch");
		}

		T sum = T{};
		const T epsilon = static_cast<T>(1e-7);

		size_t count = predictions.NumElements();
		for (size_t i = 0; i < count; ++i) {
			T p = std::clamp(predictions[i], epsilon, 1 - epsilon);
			sum += -targets[i] * std::log(p);
		}

		TensorCore::Tensor<T> result{ {1}, allocator };
		result[0] = sum / predictions.NumElements();

		return result;
	}
}