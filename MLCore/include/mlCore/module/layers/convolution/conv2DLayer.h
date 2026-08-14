/// conv2DLayer.h
#include <mlCore/module/module.h>
#include <mlCore/parameters/initialization.h>

namespace MLCore::NN {
	/// <summary>
	/// A templated 2D convolutional layer module that applies learnable convolution kernels to input tensors.
	/// </summary>
	/// <typeparam name="T">Numeric type used for tensors and parameters (for example, float or double).</typeparam>
	template <typename T>
	class Conv2DLayer : public Module<T> {
	public:
		/// <summary>
		/// Constructs a Conv2DLayer<T> with the specified input/output channels, kernel size, stride, padding, dilation, and initialization settings. Allocates kernel and bias tensors, enables gradient tracking for them, computes fan-in/fan-out, and initializes the parameters.
		/// </summary>
		/// <typeparam name="T">Numeric type used for tensor elements and layer parameters (e.g., float or double).</typeparam>
		/// <param name="inChannels">Number of input channels.</param>
		/// <param name="outChannels">Number of output channels (filters).</param>
		/// <param name="kernelHeight">Kernel (filter) height.</param>
		/// <param name="kernelWidth">Kernel (filter) width.</param>
		/// <param name="strideH">Vertical stride (height) for the convolution.</param>
		/// <param name="strideW">Horizontal stride (width) for the convolution.</param>
		/// <param name="paddingH">Vertical padding applied to the input.</param>
		/// <param name="paddingW">Horizontal padding applied to the input.</param>
		/// <param name="dilationH">Vertical dilation (spacing) of kernel elements.</param>
		/// <param name="dilationW">Horizontal dilation (spacing) of kernel elements.</param>
		/// <param name="kernelInit">Initialization method (InitType) used to initialize the kernel tensor.</param>
		/// <param name="biasInit">Initialization method (InitType) used to initialize the bias tensor.</param>
		Conv2DLayer(size_t inChannels, size_t outChannels, size_t kernelHeight, size_t kernelWidth, 
					size_t strideH = 1, size_t strideW = 1, size_t paddingH = 0, size_t paddingW = 0, size_t dilationH = 1, size_t dilationW = 1, 
					InitType kernelInit = InitType::XavierUniform, InitType biasInit = InitType::Zero);

		/// <summary>
		/// Applies this Conv2D layer to an input tensor, performing a 2D convolution using the layer's kernel, bias, stride, padding, and dilation settings.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors (e.g., float or double) used for the input, kernel, bias, and output.</typeparam>
		/// <param name="input">The input tensor to convolve. Supplied by const reference and must have a layout and element type compatible with this layer.</param>
		/// <returns>A new TensorCore::Tensor<T> containing the result of the 2D convolution. The method is const and does not modify the layer's stored parameters.</returns>
		virtual TensorCore::Tensor<T> Forward(const TensorCore::Tensor<T>& input) const override;

	protected:
		/// <summary>
		/// Appends references to the layer's parameter objects (kernel and bias) into the provided output vector.
		/// </summary>
		/// <typeparam name="T">The numeric/data type used by the layer's parameters (e.g., float or double).</typeparam>
		/// <param name="out">A reference to a vector of reference_wrappers for NN::Parameter<T>. The function appends references to this layer's parameters (m_Kernel and m_Bias). The vector is modified in-place; ownership is not transferred.</param>
		virtual void CollectParameters(std::vector<std::reference_wrapper<NN::Parameter<T>>>& out) override;

		/// <summary>
		/// Appends references to this layer's parameters (kernel and bias) into the provided output vector.
		/// </summary>
		/// <typeparam name="T">Type of the values stored in the layer parameters (the element type used by NN::Parameter<T>).</typeparam>
		/// <param name="out">Output vector that will receive references to the layer's parameters. The function appends std::reference_wrapper<const NN::Parameter<T>> for m_Kernel and m_Bias.</param>
		virtual void CollectParameters(std::vector<std::reference_wrapper<const NN::Parameter<T>>>& out) const override;

		/// <summary>
		/// Appends the layer's named parameters (kernel and bias) to the given output vector, using the provided name as an optional prefix.
		/// </summary>
		/// <typeparam name="T">The numeric type of the layer's parameter data (for example, float or double).</typeparam>
		/// <param name="name">Optional prefix for parameter names. If non-empty, a dot is inserted before the suffix (e.g., "prefix.kernel"); if empty, the suffix alone is used (e.g., "kernel").</param>
		/// <param name="out">Reference to a vector where NamedParameter<T> entries will be appended. This function emplaces entries for the layer's "kernel" and "bias" parameters.</param>
		virtual void CollectNamedParameters(const std::string& name, std::vector<NamedParameter<T>>& out) override;

		/// <summary>
		/// Appends this Conv2DLayer's named constant parameters (kernel and bias) to the provided output vector, using an optional name prefix.
		/// </summary>
		/// <typeparam name="T">The numeric/tensor element type used by the layer (the type parameter for ConstNamedParameter).</typeparam>
		/// <param name="name">Optional prefix for parameter names. If empty, the parameter suffixes "kernel" and "bias" are used as-is; otherwise the prefix and suffix are joined with a '.' (e.g., "prefix.kernel").</param>
		/// <param name="out">Reference to a vector that will receive ConstNamedParameter<T> entries for the layer's kernel and bias; entries are appended to the vector.</param>
		virtual void CollectNamedParameters(const std::string& name, std::vector<ConstNamedParameter<T>>& out) const override;

	private:
		Parameter<T> m_Kernel; /// Member variable that holds a kernel parameter as a Parameter<T> instance.
		Parameter<T> m_Bias; /// Member variable that holds a bias parameter as a Parameter<T> instance.

		size_t m_StrideH, m_StrideW; /// Stride along height and width dimensions for the convolution operation.

		size_t m_PaddingH, m_PaddingW; /// Padding along height and width dimensions for the convolution operation.

		size_t m_DilationH, m_DilationW; /// Dilation along height and width dimensions for the convolution operation.
	};
}

#include "conv2DLayer.inl"