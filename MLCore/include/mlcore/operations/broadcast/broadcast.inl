// broadcast.inl
#include <mlCore/autograd/functions/broadcast/broadcastGradFn.h>

namespace MLCore::Operations {
	template <typename T>
	TensorCore::Tensor<T> Squeeze(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator) {
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
	TensorCore::Tensor<T> Unsqueeze(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator) {
		if (axis > A.Rank()) {
			throw std::runtime_error("ERROR: Squeeze: Axis out of bounds");
		}

		if (A.Dims()[axis] != 1) {
			throw std::runtime_error("ERROR: Squeeze: Can only squeeze dimensions of size 1");
		}

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