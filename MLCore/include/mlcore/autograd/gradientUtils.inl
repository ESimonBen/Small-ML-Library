// gradientUtils.inl
#include <mlCore/operations/broadcast/broadcast.h>
#include <mlCore/operations/reduction/reduction.h>

namespace MLCore::AutoGrad {
	template <typename T>
	TensorCore::Tensor<T> ReduceSumToShape(const TensorCore::Tensor<T>& gradient, const Utils::Shape& targetShape) {
		#ifdef ML_CORE_DEBUG
			if (!Operations::CanBroadcast(targetShape, gradient.GetShape())) {
				throw std::runtime_error("ERROR: ReduceSumToShape: Invalid broadcast reduction");
			}
		#endif

		const auto& gradShape = gradient.GetShape();
		TensorCore::Tensor<T> grad = gradient.Detach();
		auto& allocator = grad.GetAllocator();

		TensorCore::Tensor<T> result = gradient.Clone();

		size_t gradRank = gradShape.Rank();
		size_t targetRank = targetShape.Rank();

		while (gradRank > targetRank) {
			result = std::move(Operations::AxisSum(result, 0, allocator));
			--gradRank;
		}

		// Backwards iteration to prevent invalid index access
		for (size_t i = targetRank; i-- > 0;) {
			size_t gradDim = result.Dims()[i];
			size_t targetDim = targetShape.Dims()[i];

			if (targetDim == 1 && gradDim > 1) {
				result = std::move(Operations::AxisSum(result, i, allocator));
			}
		}

		result.SetRequiresGrad(false);
		return result;
	}
	
	template <typename T>
	TensorCore::Tensor<T> ExpandToShape(const TensorCore::Tensor<T>& gradient, const Utils::Shape& targetShape) {
		auto info = Operations::ComputeBroadcastTo(gradient.GetShape(), targetShape);

		TensorCore::Tensor<T> grad = gradient.Detach();
		TensorCore::Tensor<T> output{ targetShape, grad.GetAllocator() };

		const size_t size = output.NumElements();
		auto& targetStrides = targetShape.Strides();

		for (size_t i = 0; i < size; ++i) {
			size_t idxInput = 0;
			size_t temp = i;

			size_t targetRank = targetShape.Rank();

			for (size_t j = 0; j < targetRank; ++j) {
				size_t dimIndex = temp / targetStrides[j];
				temp %= targetStrides[j];

				idxInput += dimIndex * info.strideA[j];
			}

			output[i] = gradient[idxInput];
		}

		return output;
	}
}