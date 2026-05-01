// reduction.inl
#include <limits>
#include <concepts>
#include <stdexcept>
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/autograd/functions/reduction/reductionGradFn.h>

namespace MLCore::Operations {
	template <typename T>
	inline TensorCore::Tensor<T> SumAll(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
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
			result.SetGradFn(std::make_shared<AutoGrad::SumGradFn<T>>(A.GetImpl()));
		}

		return result;
	}

	template <typename T>
	inline TensorCore::Tensor<T> MeanAll(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		static_assert(std::is_floating_point_v<T>, "ERROR: T must be a floating point type");

		size_t size = A.NumElements();

		if (size == 0) {
			throw std::runtime_error("ERROR: Mean: Tensor was empty");
		}

		TensorCore::Tensor<T> result = DivideScalar(SumAll(A, allocator), static_cast<T>(size), allocator, false);

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
		}

		return result;
	}

	template <typename T>
	inline TensorCore::Tensor<T> MaxAll(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		static_assert(std::totally_ordered<T>, "ERROR: T must be totally ordered");

		const size_t size = A.NumElements();
		if (size == 0) {
			throw std::runtime_error("ERROR: Min: Tensor is empty");
		}

		T max = A[0];

		TensorCore::Tensor<T> result{ {1}, allocator };

		for (size_t i = 1; i < size; ++i) {
			max = (max > A[i]) ? max : A[i];
		}

		result[0] = max;

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::MaxGradFn<T>>(A.GetImpl(), max));
		}

		return result;
	}

	template <typename T>
	inline TensorCore::Tensor<T> MinAll(const TensorCore::Tensor<T>& A, Memory::ArenaAllocator& allocator) {
		static_assert(std::totally_ordered<T>, "ERROR: T must be totally ordered");

		const size_t size = A.NumElements();
		if (size == 0) {
			throw std::runtime_error("ERROR: Min: Tensor is empty");
		}

		T min = A[0];

		TensorCore::Tensor<T> result{ {1}, allocator };

		for (size_t i = 1; i < size; ++i) {
			min = (min < A[i]) ? min : A[i];
		}

		result[0] = min;

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::MinGradFn<T>>(A.GetImpl(), min));
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> AxisSum(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator, bool keepDims) {
		if (axis >= A.Rank()) {
			throw std::out_of_range("ERROR: AxisSum: Axis out of bounds");
		}

		const std::vector<size_t>& dims = A.Dims();
		size_t rank = A.Rank();

		// This is assuming that keepDims is fault, and needs to be changed for all axis-based and reduction operations/functions
		std::vector<size_t> outDims = dims;

		if (keepDims) {
			outDims[axis] = 1;
		}
		else {
			outDims.erase(outDims.begin() + axis);

			if (outDims.empty()) {
				outDims.push_back(1);
			}
		}


		TensorCore::Tensor<T> result{ outDims, allocator };
		result.Fill(static_cast<T>(0)); // Temporary fix

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

		std::vector<size_t> outCoords(result.Rank());

		for (size_t o = 0; o < outer; ++o) {
			for (size_t i = 0; i < inner; ++i) {
				size_t base = o * axisSize * inner + i;

				T sum = static_cast<T>(0);

				for (size_t j = 0; j < axisSize; ++j) {
					sum += A[base + j * inner];
				}

				/*result[o * inner + i] = sum;*/

				// Let's see if this works
				size_t tmp = o;

				for (int b = ((int)axis) - 1; b >= 0; --b) {
					outCoords[b] = tmp % dims[b];
					tmp /= dims[b];
				}

				if (keepDims) {
					outCoords[axis] = 0;
				}

				tmp = i;

				for (int b = ((int)rank) - 1; b > (int)axis; --b) {
					size_t outIdx = (keepDims) ? b : b - 1;
					outCoords[outIdx] = tmp % dims[b];
					tmp /= dims[b];
				}

				result(outCoords) = sum;
			}
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::AxisSumGradFn<T>>(A.GetImpl(), axis, keepDims));
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> AxisMean(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator, bool keepDims) {
		if (axis >= A.Rank()) {
			throw std::out_of_range("ERROR: AxisMean: Axis out of bounds");
		}

		size_t axisSize = A.Dims()[axis];

		TensorCore::Tensor<T> result = DivideScalar(AxisSum(A, axis, allocator, keepDims), static_cast<T>(axisSize), allocator, false);

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> AxisMax(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator, bool keepDims) {
		if (axis >= A.Rank()) {
			throw std::out_of_range("ERROR: AxisMax: Axis out of bounds");
		}

		const std::vector<size_t>& dims = A.Dims();
		size_t rank = A.Rank();

		std::vector<size_t> outDims = dims;

		if (keepDims) {
			outDims[axis] = 1;
		}
		else {
			outDims.erase(outDims.begin() + axis);

			if (outDims.empty()) {
				outDims.push_back(1);
			}
		}

		TensorCore::Tensor<T> result{ outDims, allocator };
		result.Fill(static_cast<T>(0)); // Temporary fix

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

				result[o * inner + i] = max;
			}
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::AxisMaxGradFn<T>>(A.GetImpl(), axis, keepDims));
		}

		return result;
	}

	template <typename T>
	TensorCore::Tensor<T> AxisMin(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator, bool keepDims) {
		if (axis >= A.Rank()) {
			throw std::out_of_range("ERROR: AxisMin: Axis out of bounds");
		}

		const std::vector<size_t>& dims = A.Dims();
		size_t rank = A.Rank();

		std::vector<size_t> outDims = dims;
		
		if (keepDims) {
			outDims[axis] = 1;
		}
		else {
			outDims.erase(outDims.begin() + axis);

			if (outDims.empty()) {
				outDims.push_back(1);
			}
		}

		TensorCore::Tensor<T> result{ outDims, allocator };
		result.Fill(static_cast<T>(0)); // Temporary fix

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

				T min = std::numeric_limits<T>::infinity();

				for (size_t j = 0; j < axisSize; ++j) {
					T testVal = A[base + j * inner];
					min = (min < testVal) ? min : testVal;
				}

				result[o * inner + i] = min;
			}
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::AxisMinGradFn<T>>(A.GetImpl(), axis, keepDims));
		}

		return result;
	}
}