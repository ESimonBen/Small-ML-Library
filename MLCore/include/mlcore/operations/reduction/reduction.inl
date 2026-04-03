// reduction.inl
#include <concepts>
#include <stdexcept>

namespace MLCore::Operations {
	template <typename T>
	inline T Sum(const TensorCore::Tensor<T>& A) {
		static_assert(std::is_arithmetic_v<T>, "ERROR: T must be an arithmetic type");

		const size_t size = A.NumElements();

		if (size == 0) {
			return 0;
		}

		const T* data = A.Data();

		T result = data[0];

		for (size_t i = 1; i < size; ++i) {
			result += data[i];
		}

		return result;
	}

	template <typename T>
	inline T Mean(const TensorCore::Tensor<T>& A) {
		static_assert(std::is_floating_point_v<T>, "ERROR: T must be a floating point type");
		return Sum(A) / (static_cast<T>(A.NumElements()));
	}

	template <typename T>
	inline T Max(const TensorCore::Tensor<T>& A) {
		static_assert(std::totally_ordered<T>, "T must be totally ordered");

		const size_t size = A.NumElements();
		if (size == 0) {
			throw std::runtime_error("ERROR: Tensor is empty");
		}

		const T* data = A.Data();

		T result = data[0];

		for (size_t i = 1; i < size; ++i) {
			result = (result > data[i]) ? result : data[i];
		}

		return result;
	}

	template <typename T>
	inline T Min(const TensorCore::Tensor<T>& A) {
		static_assert(std::totally_ordered<T>, "T must be totally ordered");

		const size_t size = A.NumElements();
		if (size == 0) {
			throw std::runtime_error("ERROR: Tensor is empty");
		}

		const T* data = A.Data();

		T result = data[0];

		for (size_t i = 1; i < size; ++i) {
			result = (result < data[i]) ? result : data[i];
		}

		return result;
	}
}