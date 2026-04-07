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

		T result = A[0];

		for (size_t i = 1; i < size; ++i) {
			result += A[i];
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

		T result = A[0];

		for (size_t i = 1; i < size; ++i) {
			result = (result > A[i]) ? result : A[i];
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

		T result = A[0];

		for (size_t i = 1; i < size; ++i) {
			result = (result < A[i]) ? result : A[i];
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> AxisSum(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator) {
		if (axis >= A.Rank()) {
			throw std::out_of_range("ERROR: Sum: Axis out of bounds");
		}

		std::vector<size_t> dims = A.Dims();
		dims.erase(dims.begin() + axis);

		if (dims.empty()) {
			dims.push_back(1);
		}

		TensorCore::Tensor<T> result{ dims, allocator };

		for (size_t i = 0; i < result.NumElements(); ++i) {
			std::vector<size_t> indices = result.GetShape().UnflattenIndex(i); // Can be optimized later
			T sum = T{};

			for (size_t j = 0; j < A.Dims()[axis]; ++j) {
				std::vector<size_t> fullIndices = indices; // Can be optimized later
				fullIndices.insert(fullIndices.begin() + axis, j);
				sum += A[A.GetShape().FlattenIndex(fullIndices)];
			}

			result[i] = sum;
		}

		return result;
	}
}