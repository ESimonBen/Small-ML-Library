/// convolution.inl
#include <mlCore/autograd/functions/convolution/convolutionGradFn.h>

namespace MLCore::Operations {
	template <typename T>
	inline TensorCore::Tensor<T> Conv2D(const TensorCore::Tensor<T>& input, const TensorCore::Tensor<T>& kernel, const TensorCore::Tensor<T>* bias,
										size_t strideH, size_t strideW, size_t paddingH, size_t paddingW, size_t dilationH, size_t dilationW) {
		if (&input.GetAllocator() != &kernel.GetAllocator()) {
			throw std::runtime_error("ERROR: Operations between tensors on different allocators are forbidden");
		}

		if (bias && &input.GetAllocator() != &bias->GetAllocator()) {
			throw std::runtime_error("ERROR: Operations between tensors on different allocators are forbidden");
		}
		
		if (input.Rank() != 4) {
			throw std::runtime_error("ERROR: Conv2D: Input must have 4 dimensions");
		}

		if (kernel.Rank() != 4) {
			throw std::runtime_error("ERROR: Conv2D: kernel must have 4 dimensions");
		}
		
		const size_t batchSize = input.Dims()[0];
		const size_t inputChannels = input.Dims()[1];
		const size_t inputHeight = input.Dims()[2];
		const size_t inputWidth = input.Dims()[3];

		const size_t outputChannels = kernel.Dims()[0];
		const size_t kernelChannels = kernel.Dims()[1];
		const size_t kernelHeight = kernel.Dims()[2];
		const size_t kernelWidth = kernel.Dims()[3];

		if (inputChannels != kernelChannels) {
			throw std::runtime_error("ERROR: Conv2D: Input channels do not match kernel channels");
		}

		if (strideH == 0 || strideW == 0) {
			throw std::runtime_error("ERROR: Conv2D: Stride cannot be 0");
		}

		if (dilationH == 0 || dilationW == 0) {
			throw std::runtime_error("ERROR: Conv2D: Dilation cannot be 0");
		}

		if (kernelHeight == 0 || kernelWidth == 0) {
			throw std::runtime_error("ERROR: Conv2D: Kernel dimensions cannot be 0");
		}

		const size_t effectiveKernelHeight = dilationH * (kernelHeight - 1) + 1;
		const size_t effectiveKernelWidth = dilationW * (kernelWidth - 1) + 1;

		const size_t paddedInputHeight = inputHeight + 2 * paddingH;
		const size_t paddedIputWidth = inputWidth + 2 * paddingW;

		if (paddedInputHeight < effectiveKernelHeight || paddedIputWidth < effectiveKernelWidth) {
			throw std::runtime_error("ERROR: Conv2D: Kernel is larger than padded input");
		}

		const size_t outputHeight = (paddedInputHeight - effectiveKernelHeight) / strideH + 1;
		const size_t outputWidth = (paddedIputWidth - effectiveKernelWidth) / strideW + 1;

		Memory::ArenaAllocator& allocator = input.GetAllocator();

		TensorCore::Tensor<T> output{{batchSize, outputChannels, outputHeight, outputWidth}, allocator};

		for (size_t n = 0; n < batchSize; ++n) {
			for (size_t oc = 0; oc < outputChannels; ++oc) {
				for (size_t oh = 0; oh < outputHeight; ++oh) {
					for (size_t ow = 0; ow < outputWidth; ++ow) {
						T sum = static_cast<T>(0);

						for (size_t ic = 0; ic < inputChannels; ++ic) {
							for (size_t kh = 0; kh < kernelHeight; ++kh) {
								for (size_t kw = 0; kw < kernelWidth; ++kw) {
									const int inputRow = static_cast<int>(oh * strideH) + static_cast<int>(kh * dilationH) - static_cast<int>(paddingH);
									const int inputCol = static_cast<int>(ow * strideW) + static_cast<int>(kw * dilationW) - static_cast<int>(paddingW);

									if (inputRow < 0 || inputRow >= static_cast<int>(inputHeight) || inputCol < 0 || inputCol >= static_cast<int>(inputWidth)) {
										continue;
									}

									const size_t inputIndex = ((n * inputChannels + ic) * inputHeight + static_cast<size_t>(inputRow)) * inputWidth + static_cast<size_t>(inputCol);
									const size_t kernelIndex = ((oc * inputChannels + ic) * kernelHeight + kh) * kernelWidth + kw;

									sum += input[inputIndex] * kernel[kernelIndex];
								}
							}
						}

						if (bias) {
							sum += (*bias)[oc];
						}

						const size_t outputIndex = ((n * outputChannels + oc) * outputHeight + oh) * outputWidth + ow;
						output[outputIndex] = sum;
					}
				}
			}
		}

		if (input.RequiresGrad() || kernel.RequiresGrad() || (bias && bias->RequiresGrad())) {
			output.SetRequiresGrad(true);
			output.SetGradFn(std::make_shared<AutoGrad::Conv2DGradFn<T>>(input.GetImpl(), kernel.GetImpl(), bias ? bias->GetImpl() : nullptr,
																		 strideH, strideW, paddingH, paddingW, dilationH, dilationW));
		}

		return output;
	}
}