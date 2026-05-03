// initialization.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Init {
	enum class InitType {
		XavierUniform,
		XavierNormal,
		HeUniform,
		HeNormal,
		Zero
	};

	template <typename T>
	void Init(TensorCore::Tensor<T>& tensor, size_t fan_in, size_t fan_out, InitType type);
}

#include "initialization.inl"