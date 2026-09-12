/// broadcastingLayers.h
#pragma once
#include <mlCore/module/module.h>

namespace MLCore::NN {
	/// <summary>
	/// A layer that reshapes an input tensor to a predefined target shape.
	/// </summary>
	/// <typeparam name="T">The element type of tensors processed by the layer (for example, float or double).</typeparam>
	template <typename T>
	class ReshapeLayer : public Module<T> {
	public:
		/// <summary>
		/// Constructs a ReshapeLayer configured with the specified target shape.
		/// </summary>
		/// <typeparam name="T">The element type used by the layer (e.g., float, double, or a custom numeric type).</typeparam>
		/// <param name="targetShape">The desired output shape for the layer; stored as the layer's target shape.</param>
		ReshapeLayer(const Utils::Shape& targetShape);

		/// <summary>
		/// Reshapes the input tensor to the layer's target shape. If the input is rank 1, it is reshaped directly to the target shape; otherwise the input's batch dimension (first dimension) is preserved and the remaining dimensions are set to the target shape.
		/// </summary>
		/// <typeparam name="T">Element type of the tensor (data type stored in the tensor).</typeparam>
		/// <param name="input">The input tensor to reshape. Passed by const reference. If input.Rank() == 1 the tensor is reshaped directly to m_TargetShape; otherwise the batch dimension (input.Dims()[0]) is kept and the rest of the dimensions are replaced by m_TargetShape.</param>
		/// <returns>A TensorCore::Tensor containing the reshaped tensor. For rank-1 input, the result has m_TargetShape; for higher-rank input, the result has the input batch dimension followed by m_TargetShape.</returns>
		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) override;

	private:
		const Utils::Shape m_TargetShape; /// Target shape for the input to be reshaped to.
	};
	
	/// <summary>
	/// A neural network layer that flattens its input tensor, collapsing spatial or feature dimensions into a single feature axis while typically preserving the batch dimension.
	/// </summary>
	/// <typeparam name="T">The element type of the tensor (e.g., float, double, or a custom numeric type).</typeparam>
	template <typename T>
	class FlattenLayer : public Module<T> {
	public:
		/// <summary>
		/// Default constructor for FlattenLayer that performs default initialization.
		/// </summary>
		FlattenLayer() = default;

		/// <summary>
		/// Performs the forward pass of a FlattenLayer by returning a flattened version of the input tensor.
		/// </summary>
		/// <typeparam name="T">The element type of the tensor (e.g., float, double, int).</typeparam>
		/// <param name="input">The input tensor to be flattened (passed as a const reference).</param>
		/// <returns>A TensorCore::Tensor containing the flattened representation of the input tensor.</returns>
		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) override;
	};

	/// <summary>
	/// A module layer that unflattens (reshapes) a tensor by expanding a specified dimension into a provided inner shape.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensor (e.g., float, double, int).</typeparam>
	template <typename T>
	class UnflattenLayer : public Module<T> {
	public:
		/// <summary>
		/// Constructs an UnflattenLayer that will unflatten the specified dimension into the provided inner shape.
		/// </summary>
		/// <typeparam name="T">The value type used by the layer (e.g., the element type for tensors or numeric operations).</typeparam>
		/// <param name="dim">The index of the dimension to unflatten.</param>
		/// <param name="innerShape">The shape to expand the specified dimension into; stored in the layer's m_InnerShape.</param>
		UnflattenLayer(size_t dim, const Utils::Shape& innerShape);

		/// <summary>
		/// Performs the forward pass of an UnflattenLayer by unflattening (reshaping) the input tensor along the layer's configured dimension into the layer's inner shape.
		/// </summary>
		/// <typeparam name="T">The element type of the tensor (e.g., float, double, int).</typeparam>
		/// <param name="input">The input tensor to be unflattened (reshaped). Passed by const reference.</param>
		/// <returns>A TensorCore::Tensor containing the unflattened (reshaped) result. The output shape is determined by the layer's m_Dim and m_InnerShape configuration.</returns>
		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) override;

	private:
		size_t m_Dim; /// Dimension to be unflattened
		const Utils::Shape m_InnerShape; /// Shape for the dimension to be unflattened into
	};
}

#include "broadcastingLayers.inl"