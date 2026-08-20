/// strides.h
#pragma once
#include <mlCore/utils/shape.h>

namespace MLCore::Utils{
	inline std::vector<size_t> ComputeContiguousStrides(const Shape& shape) {
		const std::vector<size_t>& dims = shape.Dims();
		std::vector<size_t> strides(dims.size());

		if (!dims.empty()) {
			strides.back() = 1;
			const size_t dimsSize = dims.size();

			for (size_t i = dimsSize - 1; i > 0 ; --i) {
				strides[i - 1] = strides[i] * dims[i];
			}
		}

		return strides;
	}
}