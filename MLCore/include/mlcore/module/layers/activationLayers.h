 /// activationLayers.h
#pragma once
#include <mlCore/module/module.h>

namespace MLCore::NN {
	/// <summary>
	/// A ReLU (Rectified Linear Unit) layer that applies an element-wise ReLU activation to an input tensor.
	/// </summary>
	/// <typeparam name="T">The numeric element type of the tensor (for example float or double).</typeparam>
	template <typename T>
	class ReLULayer : public Module<T> {
	public:
		/// <summary>
		/// Default constructor for ReLULayer that performs default initialization.
		/// </summary>
		ReLULayer() = default;

		/// <summary>
		/// Applies the ReLU activation function to the input tensor and returns a new tensor with the result, using the input tensor's allocator for output allocation.
		/// </summary>
		///	<typeparam name="T">Element type of the tensor (for example float or double).</typeparam>
		/// <param name="input">Const reference to the input tensor whose elements will be transformed by ReLU. The function does not modify the input and uses input.GetAllocator() to allocate the result.</param>
		/// <returns>A TensorCore::Tensor<T> containing the element-wise ReLU of the input tensor.</returns>
		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const override;
	};

	/// <summary>
	/// A templated Leaky ReLU activation layer that applies the leaky rectified linear unit to input tensors and stores a negative-slope parameter.
	/// </summary>
	/// <typeparam name="T">The numeric type used for tensor elements (for example, float or double).</typeparam>
	template <typename T>
	class LeakyReLULayer : public Module<T> {
	public:
		/// <summary>
		/// Initializes a LeakyReLULayer with the specified negative-slope coefficient.
		/// </summary>
		/// <typeparam name="T">Numeric type used for the layer's parameters and computations (for example, float or double).</typeparam>
		/// <param name="alpha">Leakiness coefficient (slope applied to negative inputs). Must be non-negative; the constructor asserts that alpha >= 0.</param>
		LeakyReLULayer(T alpha = static_cast<T>(.01));

		/// <summary>
		/// Applies the layer's Leaky ReLU activation to the input tensor and returns the resulting tensor.
		/// </summary>
		/// <typeparam name="T">The element type of the tensor (e.g., float, double) used for the tensor's values.</typeparam>
		/// <param name="input">Const reference to the input tensor whose elements will be transformed; the tensor's allocator is used for the output allocation.</param>
		/// <returns>A new TensorCore::Tensor<T> containing the element-wise Leaky ReLU output (x >= 0 ? x : m_Alpha * x), allocated with the input's allocator.</returns>
		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const override;

		/// <summary>
		/// Returns the layer's alpha (negative slope) parameter.
		/// </summary>
		/// <typeparam name="T">The numeric type used by the layer (e.g., float, double).</typeparam>
		/// <returns>The alpha value (negative slope) for the LeakyReLU layer, of type T.</returns>
		T Alpha() const;

	private:
		const T m_Alpha; /// Slope for negative input values (multiplier applied to elements <= 0).
	};

	/// <summary>
	/// Layer that applies the hyperbolic tangent (tanh) activation elementwise to an input tensor.
	/// </summary>
	/// <typeparam name="T">The numeric type of the tensor elements (e.g., float, double).</typeparam>
	template <typename T>
	class TanhLayer : public Module<T> {
	public:
		/// <summary>
		/// Default constructor that default-initializes a TanhLayer instance.
		/// </summary>
		TanhLayer() = default;

		/// <summary>
		/// Applies the hyperbolic tangent (tanh) activation to the input tensor and returns the resulting tensor.
		/// </summary>
		/// <typeparam name="T">The element type of the tensor (for example, float or double).</typeparam>
		/// <param name="input">The input tensor whose elements will be transformed by tanh. Passed as a const reference; the input's allocator is used for any required allocations.</param>
		/// <returns>A TensorCore::Tensor<T> containing the element-wise tanh of the input tensor, allocated using the input's allocator.</returns>
		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const override;
	};

	/// <summary>
	/// A layer that applies the sigmoid activation function element-wise to an input tensor. The Forward method computes the sigmoid of each element and returns the resulting tensor. Note: Do NOT use this with BCEWithLogits, as that already uses the Sigmoid activation function internally
	/// </summary>
	/// <typeparam name="T">The numeric type of the tensor elements (for example, float or double).</typeparam>
	template <typename T>
	class SigmoidLayer : public Module<T> {
	public:
		/// <summary>
		/// Default constructor for SigmoidLayer that performs default member initialization.
		/// </summary>
		SigmoidLayer() = default;

		/// <summary>
		/// Applies the sigmoid activation element-wise to the input tensor and returns the resulting tensor.
		/// </summary>
		/// <typeparam name="T">The element type of the tensor (e.g., float, double) used for computations.</typeparam>
		/// <param name="input">The input tensor to which the sigmoid activation is applied. The allocator associated with this tensor is used for allocating the output.</param>
		/// <returns>A tensor of type TensorCore::Tensor<T> containing the element-wise sigmoid of the input, allocated using the input tensor's allocator.</returns>
		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const override;
	};
}
#include "activationLayers.inl"