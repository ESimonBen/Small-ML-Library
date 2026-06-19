 /// loss.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Operations {
	/// <summary>
	/// Specifies how values should be reduced or aggregated.
	/// </summary>
	enum class Reduction {
		None,
		Mean,
		Sum
	};

	/// <summary>
	/// Computes the mean squared error (MSE) between predictions and targets along the specified axis and returns either per-sample MSE or a reduced result according to the provided reduction mode.
	/// </summary>
	/// <typeparam name="T">Element type of the input and output tensors (typically a floating-point numeric type).</typeparam>
	/// <param name="predictions">Tensor of predicted values. Must have the same shape as targets.</param>
	/// <param name="targets">Tensor of ground-truth values. Must have the same shape as predictions.</param>
	/// <param name="axis">Axis over which to compute the per-sample mean of the squared errors. Must be less than predictions.Rank().</param>
	/// <param name="config">Reduction mode to apply to the per-sample MSE: Reduction::None returns the per-sample MSE tensor, Reduction::Mean returns the mean over all per-sample MSE values, and Reduction::Sum returns the sum of all per-sample MSE values.</param>
	/// <param name="allocator">Allocator used to allocate intermediate and result tensors.</param>
	/// <returns>A Tensor<T> containing the MSE. If config is Reduction::None, returns the per-sample MSE along the specified axis; if Reduction::Mean or Reduction::Sum, returns the corresponding scalar or reduced tensor containing the aggregated result.</returns>
	template <typename T>
	TensorCore::Tensor<T> MeanSquaredError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the mean absolute error (MAE) between predictions and targets along a specified axis, with a configurable reduction (None, Mean, or Sum). Throws on shape mismatch, invalid axis, or unsupported reduction.
	/// </summary>
	/// <typeparam name="T">Element type of the tensors (for example float or double).</typeparam>
	/// <param name="predictions">Tensor of predicted values. Must have the same shape as targets.</param>
	/// <param name="targets">Tensor of ground-truth values. Must have the same shape as predictions.</param>
	/// <param name="axis">The axis along which to compute the mean of absolute differences (per-sample reduction axis). Must be less than predictions.Rank().</param>
	/// <param name="config">Reduction mode specifying the final aggregation: Reduction::None returns per-sample MAE, Reduction::Mean returns the mean of those values, Reduction::Sum returns their sum.</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate intermediate and result tensors.</param>
	/// <returns>A Tensor<T> containing the result: if config is Reduction::None, a tensor of per-sample MAE values (reduced along the given axis); if config is Reduction::Mean or Reduction::Sum, a scalar tensor holding the mean or sum of the per-sample MAEs respectively. Throws std::runtime_error on shape mismatch or invalid reduction, and std::out_of_range if axis is out of bounds.</returns>
	template <typename T>
	TensorCore::Tensor<T> MeanAbsoluteError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the binary cross-entropy loss between prediction and target tensors. Predictions are clamped to avoid log(0) and the loss is reduced according to the specified Reduction mode.
	/// </summary>
	/// <typeparam name="T">Numeric element type for the tensors (e.g., float or double).</typeparam>
	/// <param name="predictions">Tensor of predicted probabilities (same shape as targets). Values are clamped to [epsilon, 1 - epsilon] internally to avoid taking log of 0.</param>
	/// <param name="targets">Tensor of target binary labels (same shape as predictions).</param>
	/// <param name="axis">Axis along which to compute the per-sample mean of the elementwise loss before applying the final reduction. Must be a valid axis for the input tensors.</param>
	/// <param name="config">Reduction mode indicating how to aggregate per-sample losses: None returns per-sample losses, Mean returns the mean over all samples, Sum returns the sum.</param>
	/// <param name="allocator">Arena allocator used for intermediate and output tensor allocations.</param>
	/// <returns>A Tensor<T> containing the loss. If config is Reduction::None, returns the per-sample loss tensor (reduced along the given axis). If config is Reduction::Mean or Reduction::Sum, returns a tensor containing the scalar mean or sum of the per-sample losses, respectively. The function throws on shape mismatch, invalid axis, or invalid reduction option.</returns>
	template <typename T>
	TensorCore::Tensor<T> BinaryCrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the binary cross-entropy loss from logits and targets using a numerically stable formulation. Returns per-sample losses along the given axis or an aggregated scalar depending on the specified reduction. Throws std::runtime_error on shape mismatch or invalid reduction and std::out_of_range if the axis is out of bounds.
	/// </summary>
	/// <typeparam name="T">Numeric element type of the tensors (for example float or double).</typeparam>
	/// <param name="logits">Tensor of predicted logits. Must have the same shape as targets.</param>
	/// <param name="targets">Tensor of target values (typically 0 or 1) with the same shape as logits.</param>
	/// <param name="axis">Axis along which to compute per-sample means before reduction. Must be less than logits.Rank().</param>
	/// <param name="config">Reduction mode (Reduction::None to return per-sample tensor, Reduction::Mean to return the mean loss, or Reduction::Sum to return the summed loss).</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate intermediate and result tensors.</param>
	/// <returns>A Tensor<T> containing per-sample losses if Reduction::None, or a scalar tensor with the aggregated loss for Reduction::Mean or Reduction::Sum.</returns>
	template <typename T>
	TensorCore::Tensor<T> BinaryCrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the cross-entropy loss between prediction and target tensors along a specified axis and applies an optional reduction. This assumes that the result of Softmax is being passed in. (Softmax returns the probabilities of each element of the prediction).
	/// </summary>
	/// <typeparam name="T">Element type of the tensors (e.g., float or double).</typeparam>
	/// <param name="predictions">Tensor of predicted probabilities with the same shape as targets.</param>
	/// <param name="targets">Tensor of target probabilities or one-hot labels with the same shape as predictions.</param>
	/// <param name="axis">Axis along which to compute the per-sample mean (must be less than the rank of predictions).</param>
	/// <param name="config">Reduction mode to apply to the per-sample losses (Reduction::None returns the per-sample tensor; Reduction::Mean or Reduction::Sum reduces to a scalar).</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate temporary tensors and the result.</param>
	/// <returns>A TensorCore::Tensor<T> containing the cross-entropy loss. If config is Reduction::None the tensor contains per-sample losses (reduced over the specified axis). If config is Reduction::Mean or Reduction::Sum the function returns a scalar tensor with the mean or sum of the per-sample losses, respectively. The function may throw runtime_error for shape mismatch or invalid reduction and out_of_range if axis is out of bounds.</returns>
	template <typename T>
	TensorCore::Tensor<T> CrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the cross-entropy loss between logits and target distributions using a numerically stable log-softmax along the specified axis. The function returns per-sample losses or an aggregated scalar depending on the Reduction mode. It throws std::runtime_error for shape mismatches or invalid reduction options and std::out_of_range if the axis is out of bounds. Intermediate allocations use the provided allocator.
	/// </summary>
	/// <typeparam name="T">Element type of the tensors (e.g., float, double).</typeparam>
	/// <param name="logits">Tensor of raw prediction scores (logits). Must have the same shape as targets.</param>
	/// <param name="targets">Tensor of target distributions or one-hot labels matching the shape of logits.</param>
	/// <param name="axis">Axis that represents the class dimension for softmax/log-softmax. Must be less than logits.Rank().</param>
	/// <param name="config">Reduction mode to apply to the per-sample losses (Reduction::None returns per-sample losses; Reduction::Mean returns the mean; Reduction::Sum returns the sum).</param>
	/// <param name="allocator">Memory::ArenaAllocator used for allocating intermediate tensors and the result.</param>
	/// <returns>A Tensor<T> containing the computed cross-entropy loss. For Reduction::None this is a tensor of per-sample losses (with the class axis reduced); for Reduction::Mean or Reduction::Sum this is a scalar tensor containing the mean or sum of the per-sample losses, respectively.</returns>
	template <typename T>
	TensorCore::Tensor<T> CrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, size_t axis, Reduction config, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the mean squared error (MSE) between predictions and targets along the specified axis and returns either per-sample MSE or a reduced result according to the provided reduction mode. (Assuming axis = last axis here)
	/// </summary>
	/// <typeparam name="T">Element type of the input and output tensors (typically a floating-point numeric type).</typeparam>
	/// <param name="predictions">Tensor of predicted values. Must have the same shape as targets.</param>
	/// <param name="targets">Tensor of ground-truth values. Must have the same shape as predictions.</param>
	/// <param name="axis">Axis over which to compute the per-sample mean of the squared errors. Must be less than predictions.Rank().</param>
	/// <param name="config">Reduction mode to apply to the per-sample MSE: Reduction::None returns the per-sample MSE tensor, Reduction::Mean returns the mean over all per-sample MSE values, and Reduction::Sum returns the sum of all per-sample MSE values.</param>
	/// <param name="allocator">Allocator used to allocate intermediate and result tensors.</param>
	/// <returns>A Tensor<T> containing the MSE. If config is Reduction::None, returns the per-sample MSE along the specified axis; if Reduction::Mean or Reduction::Sum, returns the corresponding scalar or reduced tensor containing the aggregated result.</returns>
	template <typename T>
	TensorCore::Tensor<T> MeanSquaredError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Reduction config, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the mean absolute error (MAE) between predictions and targets along a specified axis, with a configurable reduction (None, Mean, or Sum). Throws on shape mismatch, invalid axis, or unsupported reduction. (Assuming axis = last axis here)
	/// </summary>
	/// <typeparam name="T">Element type of the tensors (for example float or double).</typeparam>
	/// <param name="predictions">Tensor of predicted values. Must have the same shape as targets.</param>
	/// <param name="targets">Tensor of ground-truth values. Must have the same shape as predictions.</param>
	/// <param name="axis">The axis along which to compute the mean of absolute differences (per-sample reduction axis). Must be less than predictions.Rank().</param>
	/// <param name="config">Reduction mode specifying the final aggregation: Reduction::None returns per-sample MAE, Reduction::Mean returns the mean of those values, Reduction::Sum returns their sum.</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate intermediate and result tensors.</param>
	/// <returns>A Tensor<T> containing the result: if config is Reduction::None, a tensor of per-sample MAE values (reduced along the given axis); if config is Reduction::Mean or Reduction::Sum, a scalar tensor holding the mean or sum of the per-sample MAEs respectively. Throws std::runtime_error on shape mismatch or invalid reduction, and std::out_of_range if axis is out of bounds.</returns>
	template <typename T>
	TensorCore::Tensor<T> MeanAbsoluteError(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Reduction config, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the binary cross-entropy loss between prediction and target tensors. Predictions are clamped to avoid log(0) and the loss is reduced according to the specified Reduction mode. (Assuming axis = last axis here)
	/// </summary>
	/// <typeparam name="T">Numeric element type for the tensors (e.g., float or double).</typeparam>
	/// <param name="predictions">Tensor of predicted probabilities (same shape as targets). Values are clamped to [epsilon, 1 - epsilon] internally to avoid taking log of 0.</param>
	/// <param name="targets">Tensor of target binary labels (same shape as predictions).</param>
	/// <param name="axis">Axis along which to compute the per-sample mean of the elementwise loss before applying the final reduction. Must be a valid axis for the input tensors.</param>
	/// <param name="config">Reduction mode indicating how to aggregate per-sample losses: None returns per-sample losses, Mean returns the mean over all samples, Sum returns the sum.</param>
	/// <param name="allocator">Arena allocator used for intermediate and output tensor allocations.</param>
	/// <returns>A Tensor<T> containing the loss. If config is Reduction::None, returns the per-sample loss tensor (reduced along the given axis). If config is Reduction::Mean or Reduction::Sum, returns a tensor containing the scalar mean or sum of the per-sample losses, respectively. The function throws on shape mismatch, invalid axis, or invalid reduction option.</returns>
	template <typename T>
	TensorCore::Tensor<T> BinaryCrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Reduction config, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the binary cross-entropy loss from logits and targets using a numerically stable formulation. Returns per-sample losses along the given axis or an aggregated scalar depending on the specified reduction. Throws std::runtime_error on shape mismatch or invalid reduction and std::out_of_range if the axis is out of bounds. (Assuming axis = last axis here)
	/// </summary>
	/// <typeparam name="T">Numeric element type of the tensors (for example float or double).</typeparam>
	/// <param name="logits">Tensor of predicted logits. Must have the same shape as targets.</param>
	/// <param name="targets">Tensor of target values (typically 0 or 1) with the same shape as logits.</param>
	/// <param name="axis">Axis along which to compute per-sample means before reduction. Must be less than logits.Rank().</param>
	/// <param name="config">Reduction mode (Reduction::None to return per-sample tensor, Reduction::Mean to return the mean loss, or Reduction::Sum to return the summed loss).</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate intermediate and result tensors.</param>
	/// <returns>A Tensor<T> containing per-sample losses if Reduction::None, or a scalar tensor with the aggregated loss for Reduction::Mean or Reduction::Sum.</returns>
	template <typename T>
	TensorCore::Tensor<T> BinaryCrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, Reduction config, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the cross-entropy loss between prediction and target tensors along a specified axis and applies an optional reduction. This assumes that the result of Softmax is being passed in. (Softmax returns the probabilities of each element of the prediction). (Assuming axis = last axis here)
	/// </summary>
	/// <typeparam name="T">Element type of the tensors (e.g., float or double).</typeparam>
	/// <param name="predictions">Tensor of predicted probabilities with the same shape as targets.</param>
	/// <param name="targets">Tensor of target probabilities or one-hot labels with the same shape as predictions.</param>
	/// <param name="axis">Axis along which to compute the per-sample mean (must be less than the rank of predictions).</param>
	/// <param name="config">Reduction mode to apply to the per-sample losses (Reduction::None returns the per-sample tensor; Reduction::Mean or Reduction::Sum reduces to a scalar).</param>
	/// <param name="allocator">Memory::ArenaAllocator used to allocate temporary tensors and the result.</param>
	/// <returns>A TensorCore::Tensor<T> containing the cross-entropy loss. If config is Reduction::None the tensor contains per-sample losses (reduced over the specified axis). If config is Reduction::Mean or Reduction::Sum the function returns a scalar tensor with the mean or sum of the per-sample losses, respectively. The function may throw runtime_error for shape mismatch or invalid reduction and out_of_range if axis is out of bounds.</returns>
	template <typename T>
	TensorCore::Tensor<T> CrossEntropy(const TensorCore::Tensor<T>& predictions, const TensorCore::Tensor<T>& targets, Reduction config, Memory::ArenaAllocator& allocator);

	/// <summary>
	/// Computes the cross-entropy loss between logits and target distributions using a numerically stable log-softmax along the specified axis. The function returns per-sample losses or an aggregated scalar depending on the Reduction mode. It throws std::runtime_error for shape mismatches or invalid reduction options and std::out_of_range if the axis is out of bounds. Intermediate allocations use the provided allocator. (Assuming axis = last axis here)
	/// </summary>
	/// <typeparam name="T">Element type of the tensors (e.g., float, double).</typeparam>
	/// <param name="logits">Tensor of raw prediction scores (logits). Must have the same shape as targets.</param>
	/// <param name="targets">Tensor of target distributions or one-hot labels matching the shape of logits.</param>
	/// <param name="axis">Axis that represents the class dimension for softmax/log-softmax. Must be less than logits.Rank().</param>
	/// <param name="config">Reduction mode to apply to the per-sample losses (Reduction::None returns per-sample losses; Reduction::Mean returns the mean; Reduction::Sum returns the sum).</param>
	/// <param name="allocator">Memory::ArenaAllocator used for allocating intermediate tensors and the result.</param>
	/// <returns>A Tensor<T> containing the computed cross-entropy loss. For Reduction::None this is a tensor of per-sample losses (with the class axis reduced); for Reduction::Mean or Reduction::Sum this is a scalar tensor containing the mean or sum of the per-sample losses, respectively.</returns>
	template <typename T>
	TensorCore::Tensor<T> CrossEntropyWithLogits(const TensorCore::Tensor<T>& logits, const TensorCore::Tensor<T>& targets, Reduction config, Memory::ArenaAllocator& allocator);

	/// Planning to add other loss functions (Mean Bias Error, Huber/Smooth Mean Absolute Error, etc.)
}

#include "loss.inl"
