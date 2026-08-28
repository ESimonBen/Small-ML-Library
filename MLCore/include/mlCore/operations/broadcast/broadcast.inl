 /// broadcast.inl
#include <stdexcept>
#include <algorithm>
#include <mlCore/operations/reduction/reduction.h>
#include <mlCore/autograd/functions/broadcast/broadcastGradFn.h>

namespace MLCore::Operations {
	inline static size_t GetAlignedDim(const Utils::Shape& shape, size_t i, size_t offset) {
		return (i < offset) ? 1 : shape[i - offset];
	}
	
	inline BroadcastInfo ComputeBroadcast(const Utils::Shape& shapeA, const std::vector<size_t>& stridesA, 
										  const Utils::Shape& shapeB, const std::vector<size_t>& stridesB) {
		BroadcastInfo info;

		const size_t rankA = shapeA.Rank();
		const size_t rankB = shapeB.Rank();
		const size_t rank = std::max(rankA, rankB);

		info.strideA.resize(rank);
		info.strideB.resize(rank);

		std::vector<size_t> resultDims(rank);

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
	
	inline BroadcastInfo ComputeBroadcastTo(const Utils::Shape& sourceShape, const std::vector<size_t>& sourceStrides, const Utils::Shape& targetShape) {
		BroadcastInfo info;

		const size_t sourceRank = sourceShape.Rank();
		const size_t targetRank = targetShape.Rank();

		if (sourceRank > targetRank) {
			throw std::runtime_error("ERROR: Cannot broadcast to smaller shape");
		}

		const size_t offset = targetRank - sourceRank;

		info.shape = targetShape;
		info.strideA.resize(targetRank);

		for (size_t i = 0; i < targetRank; ++i) {
			size_t sourceDim = GetAlignedDim(sourceShape, i, offset);
			size_t targetDim = targetShape[i];

			if (sourceDim != targetDim && sourceDim != 1) {
				throw std::runtime_error( "ComputeBroadcastTo: incompatible shapes");
			}

			info.strideA[i] = (i < offset || sourceDim == 1) ? 0 : sourceStrides[i - offset];
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
			size_t dimA = GetAlignedDim(shapeA, i, offsetA);
			size_t dimB = GetAlignedDim(shapeB, i, offsetB);

			if (dimA != dimB && dimA != 1 && dimB != 1) {
				return false;
			}
		}

		return true;
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> Squeeze(const TensorCore::Tensor<T>& A, size_t axis) {
		if (axis >= A.Rank()) {
			throw std::out_of_range("ERROR: Squeeze: Axis out of bounds");
		}

		if (A.Dims()[axis] != 1) {
			throw std::runtime_error("ERROR: Squeeze: Can only squeeze dimensions of size 1");
		}

		std::vector<size_t> dims = A.Dims();
		std::vector<size_t> strides = A.Strides();

		dims.erase(dims.begin() + axis);
		strides.erase(strides.begin() + axis);

		if (dims.empty()) {
			dims.push_back(1);
			strides.push_back(1);
		}

		auto impl = std::make_shared<TensorCore::TensorImpl<T>>(
			Utils::Shape{ dims },
			strides,
			A.GetImpl()->storage,
			A.GetImpl()->allocator,
			A.GetImpl()->offset,
			A.RequiresGrad(),
			nullptr,
			nullptr
		);

		TensorCore::Tensor<T> result{ impl };

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::SqueezeGradFn<T>>(A.GetImpl(), axis));
		}

		return result;
	}
	
	template <typename T>
	inline TensorCore::Tensor<T> Unsqueeze(const TensorCore::Tensor<T>& A, size_t axis) {
		if (axis > A.Rank()) {
			throw std::out_of_range("ERROR: Unsqueeze: Axis out of bounds");
		}

		std::vector<size_t> dims = A.Dims();
		std::vector<size_t> strides = A.Strides();

		dims.insert(dims.begin() + axis, 1);
		size_t insertedStride = 1;

		if (A.Rank() == 0 || axis == A.Rank()) {
			insertedStride = 1;
		}
		else {
			insertedStride = A.Strides()[axis] * A.Dims()[axis];
		}

		strides.insert(strides.begin() + axis, insertedStride);

		auto impl = std::make_shared<TensorCore::TensorImpl<T>>(
			Utils::Shape{dims},
			strides,
			A.GetImpl()->storage,
			A.GetImpl()->allocator,
			A.GetImpl()->offset,
			A.RequiresGrad(),
			nullptr,
			nullptr
		);

		TensorCore::Tensor<T> result{ impl };

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::UnsqueezeGradFn<T>>(A.GetImpl(), axis));
		}

		return result;
	}

	template <typename T>
	inline TensorCore::Tensor<T> ReduceSumToShape(const TensorCore::Tensor<T>& A, const Utils::Shape& targetShape) {
		if (A.IsEmpty()) {
			throw std::runtime_error("ERROR: ReduceSumToShape: Input tensor cannot be null");
		}

		if (!CanBroadcast(targetShape, A.GetShape())) {
			throw std::runtime_error("ERROR: ReduceSumToShape: Invalid broadcast reduction");
		}

		if (A.Rank() < targetShape.Rank()) {
			throw std::runtime_error("ERROR: ReduceSumToShape: Target rank exceeds input rank");
		}

		TensorCore::Tensor<T> result = A.Detach();

		size_t gradRank = result.Rank();
		size_t targetRank = targetShape.Rank();

		while (gradRank > targetRank) {
			result = std::move(AxisSum(result, 0, false));
			--gradRank;
		}

		for (size_t axis = targetRank; axis-- > 0;) {
			if (targetShape[axis] == 1 && result.Dims()[axis] != 1) {
				result = AxisSum(result, axis, true);
			}
		}

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::ReduceToShapeGradFn<T>>(A.GetImpl()));
		}

		return result;
	}

	template <typename T>
	inline TensorCore::Tensor<T> ExpandToShape(const TensorCore::Tensor<T>& A, const Utils::Shape& targetShape) {
		if (A.IsEmpty()) {
			throw std::runtime_error("ERROR: ExpandToShape: Input tensor cannot be null");
		}

		auto info = ComputeBroadcastTo(A.GetShape(), A.Strides(), targetShape);

		auto impl = std::make_shared<TensorCore::TensorImpl<T>>(
			targetShape,
			info.strideA,
			A.GetImpl()->storage,
			A.GetImpl()->allocator,
			A.GetImpl()->offset,
			A.RequiresGrad(),
			nullptr,
			nullptr
		);

		TensorCore::Tensor<T> result{ impl };

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::ExpandToShapeGradFn<T>>(A.GetImpl()));
		}

		return result;
	}

	template <typename T>
	inline TensorCore::Tensor<T> Reshape(const TensorCore::Tensor<T>& A, const Utils::Shape& newShape) {
		if (A.NumElements() != newShape.NumElements()) {
			throw std::runtime_error("ERROR: Reshape: Element count mismatch");
		}

		if (!A.IsContiguous()) {
			throw std::runtime_error("ERROR: Reshape: Tensor must be contiguous");
		}

		const auto newStrides = Utils::ComputeContiguousStrides(newShape);

		auto impl = std::make_shared<TensorCore::TensorImpl<T>>(
			newShape,
			newStrides,
			A.GetImpl()->storage,
			A.GetImpl()->allocator,
			A.GetImpl()->offset,
			A.RequiresGrad(),
			nullptr,
			nullptr
		);

		TensorCore::Tensor<T> result{ impl };

		if (A.RequiresGrad()) {
			result.SetRequiresGrad(true);
			result.SetGradFn(std::make_shared<AutoGrad::ReshapeGradFn<T>>(A.GetImpl()));
		}

		return result;
	}
}