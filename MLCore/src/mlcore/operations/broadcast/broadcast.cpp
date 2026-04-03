// broadcast.cpp
#include <stdexcept>
#include <algorithm>
#include <mlCore/operations/broadcast/broadcast.h>

namespace MLCore::Operations {
	static size_t GetAlignedDim(const Utils::Shape& shape, size_t i, size_t offset) {
		return (i < offset) ? 1 : shape[i - offset];
	}

	BroadcastInfo ComputeBroadcast(const Utils::Shape& shapeA, const Utils::Shape& shapeB) {
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

	bool CanBroadcast(const Utils::Shape& shapeA, const Utils::Shape& shapeB) {
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
}