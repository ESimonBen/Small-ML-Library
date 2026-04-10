// reduction.inl
#include <concepts>
#include <stdexcept>
#include <mlCore/operations/scalar/scalar.h>

namespace MLCore::AutoGrad {
	template <typename T>
	class SumGradFn;

	template <typename T>
	class MeanGradFn;

	template <typename T>
	class MaxGradFn;

	template <typename T>
	class MinGradFn;

	template <typename T>
	class AxisSumGradFn;
}

namespace MLCore::Operations {
	template <typename T>
	inline TensorCore::Tensor<T> Sum(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		static_assert(std::is_arithmetic_v<T>, "ERROR: T must be an arithmetic type");

		const size_t size = A.NumElements();
		TensorCore::Tensor<T> result{ {1}, allocator };

		if (size == 0) {
			result[0] = static_cast<T>(0);
			return result;
		}

		T sum = static_cast<T>(0);

		for (size_t i = 0; i < size; ++i) {
			sum += A[i];
		}

		result[0] = sum;

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(new SumGradFn<T>(const_cast<TensorCore::Tensor<T>*>(&A)));
		}

		return result;
	}

	template <typename T>
	inline TensorCore::Tensor<T> Mean(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		static_assert(std::is_floating_point_v<T>, "ERROR: T must be a floating point type");

		size_t size = A.NumElements();

		if (size == 0) {
			throw std::runtime_error("ERROR: Mean: Tensor was empty");
		}

		auto sum = Sum(A, allocator);
		TensorCore::Tensor<T> result = DivideScalarRight(sum, static_cast<T>(size), allocator);

		/*if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(new MeanGradFn<T>(const_cast<TensorCore::Tensor<T>*>(&A)));
		}*/

		return result;
	}

	template <typename T>
	inline TensorCore::Tensor<T> Max(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		static_assert(std::totally_ordered<T>, "T must be totally ordered");

		const size_t size = A.NumElements();
		if (size == 0) {
			throw std::runtime_error("ERROR: Tensor is empty");
		}

		T max = A[0];

		TensorCore::Tensor<T> result{ {1}, allocator };

		for (size_t i = 1; i < size; ++i) {
			max = (max > A[i]) ? max : A[i];
		}

		result[0] = max;

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(new MaxGradFn<T>(const_cast<TensorCore::Tensor<T>*>(&A), max));
		}

		return result;
	}

	template <typename T>
	inline TensorCore::Tensor<T> Min(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		static_assert(std::totally_ordered<T>, "T must be totally ordered");

		const size_t size = A.NumElements();
		if (size == 0) {
			throw std::runtime_error("ERROR: Tensor is empty");
		}

		T min = A[0];

		TensorCore::Tensor<T> result{ {1}, allocator };

		for (size_t i = 1; i < size; ++i) {
			min = (min < A[i]) ? min : A[i];
		}

		result[0] = min;

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(new MinGradFn<T>(const_cast<TensorCore::Tensor<T>*>(&A), min));
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

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(new AxisSumGradFn<T>(const_cast<TensorCore::Tensor<T>*>(&A), axis));
		}

		return result;
	}
}