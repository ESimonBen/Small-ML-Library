// gradientUtils.inl
#include <mlCore/operations/linearAlgebra/linalg.h>

namespace MLCore::AutoGrad {
	template <typename T>
	TensorCore::Tensor<T> ReduceSumToShape(const TensorCore::Tensor<T>& gradient, const Utils::Shape& targetShape) {
		const auto& gradShape = gradient.GetShape();
		auto& allocator = const_cast<Memory::ArenaAllocator&>(gradient.GetAllocator());

		TensorCore::Tensor<T> result{ gradient };

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

		return result;
	}
}