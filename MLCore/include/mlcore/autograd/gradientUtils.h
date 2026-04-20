// gradientUtils.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::AutoGrad {
	template <typename T>
	TensorCore::Tensor<T> ReduceSumToShape(const TensorCore::Tensor<T>& gradient, const Utils::Shape& targetShape);
	
	template <typename T>
	TensorCore::Tensor<T> ExpandToShape(const TensorCore::Tensor<T>& gradient, const Utils::Shape& targetShape);
}

#include "gradientUtils.inl"