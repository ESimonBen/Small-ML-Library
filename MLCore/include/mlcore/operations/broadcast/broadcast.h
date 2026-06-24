 /// broadcast.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	/// <summary>
	/// Contains information needed to broadcast two tensors to a common shape.
	/// </summary>
	struct BroadcastInfo {
		Utils::Shape shape; /// Declares an instance of Utils::Shape.
		std::vector<size_t> strideA; /// A vector that holds stride values for array A; each element represents the stride for a corresponding dimension.
		std::vector<size_t> strideB; /// A vector that holds stride values for array B; each element represents the stride for a corresponding dimension.
	};

	/// <summary>
	/// Compute broadcast metadata for two shapes using right-aligned broadcasting rules. Determines the broadcasted shape and the per-input strides, setting strides to zero for broadcasted (size-1) dimensions.
	/// </summary>
	/// <param name="shapeA">The first input shape. Its Rank() and Strides() are used to align dimensions and compute strideA entries.</param>
	/// <param name="shapeB">The second input shape. Its Rank() and Strides() are used to align dimensions and compute strideB entries.</param>
	/// <returns>A BroadcastInfo containing the broadcasted shape (info.shape) and per-input stride vectors (info.strideA, info.strideB). The result rank is max(rankA, rankB). For any dimension where an input has size 1, that input's stride is set to 0. Throws std::runtime_error if the shapes are not compatible for broadcasting (i.e., a dimension differs and neither is 1).</returns>
	BroadcastInfo ComputeBroadcast(const Utils::Shape& shapeA, const Utils::Shape& shapeB);

	/// <summary>
	/// Computes the broadcast information required to broadcast a tensor with shape 'smaller' to the specified 'target' shape. Validates compatibility and computes per-dimension strides for the source tensor.
	/// </summary>
	/// <param name="smaller">The shape of the source (smaller) tensor to be broadcast. Its rank must be less than or equal to target.Rank().</param>
	/// <param name="target">The target shape to broadcast to.</param>
	/// <returns>A BroadcastInfo object whose 'shape' is set to target and whose 'strideA' contains per-dimension strides for the source tensor (0 for dimensions that are broadcast). The function throws std::runtime_error if the smaller shape has higher rank than target or if any dimension is incompatible for broadcasting.</returns>
	BroadcastInfo ComputeBroadcastTo(const Utils::Shape& smaller, const Utils::Shape& target);

	/// <summary>
	/// Determines whether two shapes are broadcast-compatible. The function aligns dimensions from the right (trailing dimensions) and considers a pair of dimensions compatible if they are equal or if either is 1.
	/// </summary>
	/// <param name="shapeA">The first shape to test for broadcast compatibility.</param>
	/// <param name="shapeB">The second shape to test for broadcast compatibility.</param>
	/// <returns>true if the shapes can be broadcast together under the rule that dimensions match when equal or when either is 1 (after right-aligning ranks); otherwise false.</returns>
	bool CanBroadcast(const Utils::Shape& shapeA, const Utils::Shape& shapeB);

	/// <summary>
	/// Removes the specified dimension of size 1 from the input tensor and returns a new tensor with that axis removed. Copies all elements into the result and preserves gradient metadata when present. Throws std::runtime_error if the axis is out of bounds or the selected dimension is not equal to 1.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensor.</typeparam>
	/// <param name="A">The input tensor to squeeze. Its dimension at index 'axis' must be 1.</param>
	/// <param name="axis">The index of the dimension to remove. Must refer to a valid dimension of A whose size is 1.</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate storage for the returned tensor.</param>
	/// <returns>A new TensorCore::Tensor<T> with the specified axis removed. If removing the axis results in no dimensions, the function returns a tensor with a single dimension of size 1. The returned tensor contains the same elements as the input and, if the input required gradients, the result will be marked to require gradients and will have its backward function set accordingly.</returns>
	template <typename T>
	TensorCore::Tensor<T> Squeeze(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Returns a new tensor with a size-1 dimension inserted at the specified axis, copying the elements of the input tensor and propagating gradient information when present. Throws std::runtime_error if the axis is out of range or if the checked dimension is not 1.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor.</typeparam>
	/// <param name="A">The input tensor to unsqueeze. Its contents are copied into the returned tensor.</param>
	/// <param name="axis">The position at which to insert the new dimension (valid values: 0 .. A.Rank()). Throws std::runtime_error if axis > A.Rank() or if the dimension check in the function fails.</param>
	/// <param name="allocator">Allocator used to allocate memory for the returned tensor.</param>
	/// <returns>A TensorCore::Tensor<T> with a new dimension of size 1 inserted at the specified axis, containing the same elements as A. If A.RequiresGrad() is true, the result will require gradients and will have its gradient function set appropriately.</returns>
	template <typename T>
	TensorCore::Tensor<T> Unsqueeze(const TensorCore::Tensor<T>& A, size_t axis, Memory::ArenaAllocator& allocator);
}

#include "broadcast.inl"