 /// gradientUtils.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::AutoGrad {
	/// <summary>
	/// Reduces the input gradient tensor by summing along axes until its shape matches targetShape, following broadcasting rules. Throws std::runtime_error if targetShape is not broadcast-compatible with the gradient shape.
	/// </summary>
	/// <typeparam name="T">The element type of the tensor (e.g., float, double, int).</typeparam>
	/// <param name="gradient">The input tensor to reduce. The function detaches and sums elements along axes as needed to achieve the target shape.</param>
	/// <param name="targetShape">The desired shape after reduction. Must be broadcast-compatible with gradient.GetShape(); dimensions of size 1 in targetShape cause the corresponding gradient dimensions to be summed. If targetShape has fewer ranks, leading gradient axes are summed.</param>
	/// <returns>A new tensor with shape equal to targetShape containing the reduced sums. The returned tensor has requiresGrad set to false.</returns>
	template <typename T>
	TensorCore::Tensor<T> ReduceSumToShape(const TensorCore::Tensor<T>& gradient, const Utils::Shape& targetShape);
	
	/// <summary>
	/// Broadcasts the input gradient tensor to the specified target shape and returns a new tensor containing the expanded values.
	/// </summary>
	/// <typeparam name="T">The element type stored in the input and output tensors.</typeparam>
	/// <param name="gradient">The input tensor whose values will be broadcasted. The function creates a detached copy and does not modify the original.</param>
	/// <param name="targetShape">The desired shape for the output tensor. Must be compatible with the input tensor's shape according to the broadcasting rules used by Operations::ComputeBroadcastTo.</param>
	/// <returns>A new TensorCore::Tensor<T> with shape targetShape containing values from gradient replicated according to broadcasting rules (using the same allocator as the detached input).</returns>
	template <typename T>
	TensorCore::Tensor<T> ExpandToShape(const TensorCore::Tensor<T>& gradient, const Utils::Shape& targetShape);
}

#include "gradientUtils.inl"