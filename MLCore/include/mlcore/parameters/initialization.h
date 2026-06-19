 /// initialization.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Init {
	/// <summary>
	/// Specifies common weight-initialization strategies used for neural network layers.
	/// </summary>
	enum class InitType {
		XavierUniform,
		XavierNormal,
		HeUniform,
		HeNormal,
		Zero
	};

	/// <summary>
	/// Initialize the contents of a tensor using the specified initialization strategy (zero, Xavier/Glorot, or He/Kaiming) with random values drawn from appropriate distributions.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor (e.g., float or double).</typeparam>
	/// <param name="tensor">Reference to the tensor to initialize. Its elements will be overwritten with the chosen initialization values.</param>
	/// <param name="fan_in">The number of input units used to compute initialization scale (used by Xavier and He schemes).</param>
	/// <param name="fan_out">The number of output units used to compute initialization scale (used by Xavier schemes).</param>
	/// <param name="type">Initialization method (InitType). Supported values: Zero, XavierUniform, XavierNormal, HeUniform, HeNormal.</param>
	template <typename T>
	void Init(TensorCore::Tensor<T>& tensor, size_t fan_in, size_t fan_out, InitType type);
}

#include "initialization.inl"