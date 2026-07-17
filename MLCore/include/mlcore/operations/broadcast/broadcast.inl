 /// broadcast.inl
#include <stdexcept>
#include <algorithm>
#include <mlCore/autograd/functions/broadcast/broadcastGradFn.h>

namespace MLCore::Operations {
	/// <summary>
	/// Returns the aligned dimension for a given index: if the index is before the offset it returns 1, otherwise it returns the corresponding dimension from the provided shape.
	/// </summary>
	/// <param name="shape">A reference to a Utils::Shape that provides the underlying dimension sizes.</param>
	/// <param name="i">The index in the aligned dimension space to query.</param>
	/// <param name="offset">The number of leading aligned dimensions treated as size 1; indices less than this are considered padding.</param>
	/// <returns>The dimension size for index i after alignment: 1 when i < offset, otherwise shape[i - offset].</returns>
	inline static size_t GetAlignedDim(const Utils::Shape& shape, size_t i, size_t offset) {
		return (i < offset) ? 1 : shape[i - offset];
	}
	
	inline BroadcastInfo ComputeBroadcast(const Utils::Shape& shapeA, const Utils::Shape& shapeB) {
		BroadcastInfo info;

		const size_t rankA = shapeA.Rank();
		const size_t rankB = shapeB.Rank();
		const size_t rank = std::max(rankA, rankB);

		info.strideA.resize(rank);
		info.strideB.resize(rank);

		//info.shape.resize(rank);
		std::vector<size_t> resultDims(rank);

		const auto& stridesA = shapeA.Strides();
		const auto& stridesB = shapeB.Strides();

		const size_t offsetA = rank - rankA;
		const size_t offsetB = rank - rankB;

		for (size_t i = 0; i < rank; ++i) {
			size_t dimA = GetAlignedDim(shapeA, i, offsetA);
			size_t dimB = GetAlignedDim(shapeB, i, offsetB);

			if (dimA != dimB && dimA != 1 && dimB != 1) {
				throw std::runtime_error("ERROR: Broadcast mismatch between shapes");
			}

			resultDims[i] = std::max(dimA, dimB);

			info.strideA[i] = (i < offsetA || dimA == 1) ? 0 : stridesA[i - offsetA];
			info.strideB[i] = (i < offsetB || dimB == 1) ? 0 : stridesB[i - offsetB];
		}

		info.shape = Utils::Shape{ resultDims };

		return info;
	}
	
	inline BroadcastInfo ComputeBroadcastTo(const Utils::Shape& smaller, const Utils::Shape& target) {
		BroadcastInfo info;

		const size_t smallerRank = smaller.Rank();
		const size_t targetRank = target.Rank();

		if (smallerRank > targetRank) {
			throw std::runtime_error("ERROR: Cannot broadcast to smaller shape");
		}

		info.strideA.resize(targetRank);
		info.shape = target;

		const auto& smallStrides = smaller.Strides();

		size_t offset = targetRank - smallerRank;

		for (size_t i = 0; i < targetRank; ++i) {
			size_t smallDim = GetAlignedDim(smaller, i, offset);
			size_t targetDim = target[i];

			if (smallDim != targetDim && smallDim != 1) {
				throw std::runtime_error("ERROR: ComputeBroadcastTo mismatch");
			}

			info.strideA[i] = (i < offset || smallDim == 1) ? 0 : smallStrides[i - offset];
		}

		return info;
	}
	
	inline bool CanBroadcast(const Utils::Shape& shapeA, const Utils::Shape& shapeB) {
		const size_t rankA = shapeA.Rank();
		const size_t rankB = shapeB.Rank();
		const size_t rank = std::max(rankA, rankB);

		const size_t offsetA = rank - rankA;
		const size_t offsetB = rank - rankB;

		for (size_t i = 0; i < rank; ++i) {
			size_t dimA = (i < offsetA) ? 1 : shapeA[i - offsetA];
			size_t dimB = (i < offsetB) ? 1 : shapeB[i - offsetB];

			if (dimA != dimB && dimA != 1 && dimB != 1) {
				return false;
			}
		}

		return true;
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> Squeeze(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator) {
		if (axis > A.Rank()) {
			throw std::runtime_error("ERROR: Squeeze: Axis out of bounds");
		}

		if (A.Dims()[axis] != 1) {
			throw std::runtime_error("ERROR: Squeeze: Can only squeeze dimensions of size 1");
		}

		std::vector<size_t> newDims = A.Dims();
		newDims.erase(newDims.begin() + axis);

		if (newDims.empty()) {
			newDims.push_back(1);
		}

		TensorCore::Tensor<T> result{ newDims, allocator };
		result.Fill(static_cast<T>(0));

		size_t size = A.NumElements();

		for (size_t i = 0; i < size; ++i) {
			result[i] = A[i];
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			if (A.RequiresGrad()) {
				result.SetRequiresGrad(true);
				result.SetGradFn(std::make_shared<AutoGrad::SqueezeGradFn<T>>(A.GetImpl(), axis));
			}
		}

		return result;
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> Unsqueeze(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator) {
		if (axis > A.Rank()) {
			throw std::runtime_error("ERROR: Unsqueeze: Axis out of bounds");
		}

		/*if (A.Dims()[axis] != 1) {
			throw std::runtime_error("ERROR: Unsqueeze: Can only unsqueeze dimensions of size 1");
		}*/

		std::vector<size_t> newDims = A.Dims();
		newDims.insert(newDims.begin() + axis, 1);

		TensorCore::Tensor<T> result{ newDims, allocator };
		result.Fill(static_cast<T>(0));

		size_t size = A.NumElements();

		for (size_t i = 0; i < size; ++i) {
			result[i] = A[i];
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::UnsqueezeGradFn<T>>(A.GetImpl(), axis));
		}

		return result;
	}
}