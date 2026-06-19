 /// linearLayer.h
#pragma once
#include <mlCore/module/module.h>
#include <mlCore/parameters/initialization.h>

namespace MLCore::NN {
	/// <summary>
	/// A templated fully connected (linear) neural network layer that applies an affine transformation (output = weight * input + bias), manages its parameters, and supports a forward pass.
	/// </summary>
	/// <typeparam name="T">The numeric data type used for computations and storage (e.g., float or double).</typeparam>
	template <typename T>
	class LinearLayer : public Module<T> {
	public:
		/// <summary>
		/// Constructs a LinearLayer by allocating weight and bias tensors with the given input and output sizes, enables gradient tracking for them, and initializes their values using the specified initialization types.
		/// </summary>
		/// <typeparam name="T">Numeric type of the layer's tensors (for example, float or double).</typeparam>
		/// <param name="in">Number of input features (size of the input dimension).</param>
		/// <param name="out">Number of output features (size of the output dimension).</param>
		/// <param name="allocator">Arena allocator used to allocate memory for the weight and bias tensors.</param>
		/// <param name="weightInit">Initialization type to use for the weight tensor.</param>
		/// <param name="biasInit">Initialization type to use for the bias tensor.</param>
		LinearLayer(size_t in, size_t out, Memory::ArenaAllocator& allocator, Init::InitType weightInit = Init::InitType::XavierUniform, Init::InitType biasInit = Init::InitType::Zero);

		/// <summary>
		/// Performs the forward pass of a linear (fully connected) layer: multiplies the input by the layer weights and adds the bias.
		/// </summary>
		/// <typeparam name="T">The numeric element type of the tensor (e.g., float, double) used for computation.</typeparam>
		/// <param name="input">The input tensor to the layer. The allocator from this tensor is used to allocate intermediate and output tensors. Passed by const reference.</param>
		/// <returns>A new TensorCore::Tensor<T> containing the result of the linear transformation (weight * input + bias). The tensor is allocated using the input tensor's allocator.</returns>
		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const override;

	protected:
		/// <summary>
		/// Appends references to this layer's parameter objects (weight and bias) into the provided output vector.
		/// </summary>
		/// <typeparam name="T">The numeric type used by the layer's parameters (for example, float or double).</typeparam>
		/// <param name="out">Reference to a vector that will be appended with std::reference_wrapper objects referring to the layer's parameters (m_Weight and m_Bias).</param>
		virtual void CollectParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out) override;

		/// <summary>
		/// Appends references to the layer's weight and bias parameters into the provided output vector.
		/// </summary>
		/// <typeparam name="T">The numeric type used by the layer's parameters (e.g., float, double).</typeparam>
		/// <param name="out">A vector that will receive references to the layer's parameters; the function appends references to m_Weight and m_Bias as std::reference_wrapper<const NN::Parameter<T>>.</param>
		virtual void CollectParameters(std::vector<std::reference_wrapper<const NN::Parameter<T>>>& out) const override;

		/// <summary>
		/// Appends this layer's named parameters (weight and bias) to the given output vector, using the provided name as an optional prefix.
		/// </summary>
		/// <typeparam name="T">The value type for the layer's parameters (the type stored in NamedParameter<T>, e.g., a numeric or tensor element type).</typeparam>
		/// <param name="name">Optional prefix for parameter names. If empty, parameter names are the suffixes "weight" and "bias"; otherwise the prefix and a '.' are prepended (e.g., "prefix.weight").</param>
		/// <param name="out">Reference to an output vector that will be appended with NamedParameter<T> entries referring to the layer's weight and bias.</param>
		virtual void CollectNamedParameters(const std::string& name, std::vector<NamedParameter<T>>& out) override;

		/// <summary>
		/// Appends named parameter descriptors for the layer's weight and bias to the provided output vector, using the given base name as a prefix.
		/// </summary>
		/// <typeparam name="T">Type of the layer's parameters (for example, float or double).</typeparam>
		/// <param name="name">Base name used to construct parameter names. If empty, parameters are named "weight" and "bias"; otherwise names become "<name>.weight" and "<name>.bias".</param>
		/// <param name="out">Output vector to which ConstNamedParameter<T> entries are appended. Each entry contains the generated name and a reference to the corresponding member (m_Weight or m_Bias).</param>
		virtual void CollectNamedParameters(const std::string& name, std::vector<ConstNamedParameter<T>>& out) const override;

	private:
		Parameter<T> m_Weight; /// Member variable that holds a weight parameter.
		Parameter<T> m_Bias; /// Member variable that holds a bias parameter.
	};
}

#include "linearLayer.inl"