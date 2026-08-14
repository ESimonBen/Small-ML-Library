/// convolution.inl
#include <mlCore/autograd/functions/convolution/convolutionGradFn.h>

namespace MLCore::Operations {
	template <typename T>
	inline TensorCore::Tensor<T> Conv1D(const TensorCore::Tensor<T>& input, const TensorCore::Tensor<T>& kernel, const TensorCore::Tensor<T>* bias,
										size_t stride, size_t padding, size_t dilation) {
		if (&input.GetAllocator() != &kernel.GetAllocator()) {
			throw std::runtime_error("ERROR: Operations between tensors on different allocators are forbidden");
		}

		if (bias && &input.GetAllocator() != &bias->GetAllocator()) {
			throw std::runtime_error("ERROR: Operations between tensors on different allocators are forbidden");
		}

		if (input.Rank() != 3) {
			throw std::runtime_error("ERROR: Conv1D: Input must have 3 dimensions");
		}

		if (kernel.Rank() != 3) {
			throw std::runtime_error("ERROR: Conv1D: Kernel must have 3 dimensions");
		}

		const size_t batchSize = input.Dims()[0];
		const size_t inputChannels = input.Dims()[1];
		const size_t inputLength = input.Dims()[2];

		const size_t outputChannels = kernel.Dims()[0];
		const size_t kernelChannels = kernel.Dims()[1];
		const size_t kernelLength = kernel.Dims()[2];

		if (inputChannels != kernelChannels) {
			throw std::runtime_error("ERROR: Conv1D: Input channels do not match kernel channels");
		}

		if (stride == 0) {
			throw std::runtime_error("ERROR: Conv1D: Stride cannot be 0");
		}

		if (dilation == 0) {
			throw std::runtime_error("ERROR: Conv1D: Dilation cannot be 0");
		}

		if (kernelLength == 0) {
			throw std::runtime_error("ERROR: Conv1D: Kernel dimension cannot be 0");
		}

		const size_t effectiveKernelLength = dilation * (kernelLength - 1) + 1;

		const size_t paddedInputLength = inputLength + 2 * padding;

		if (paddedInputLength < effectiveKernelLength) {
			throw std::runtime_error("ERROR: Conv1D: Kernel is larger than padded input");
		}

		const size_t outputLength = (paddedInputLength - effectiveKernelLength) / stride + 1;

		Memory::ArenaAllocator& allocator = input.GetAllocator();

		TensorCore::Tensor<T> output{ {batchSize, outputChannels, outputLength}, allocator };

		for (size_t n = 0; n < batchSize; ++n) {
			for (size_t oc = 0; oc < outputChannels; ++oc) {
				for (size_t ol = 0; ol < outputLength; ++ol) {
					T sum = static_cast<T>(0);

					for (size_t ic = 0; ic < inputChannels; ++ic) {
						for (size_t kl = 0; kl < kernelLength; ++kl) {
							const int inputPos = static_cast<int>(ol * stride) + static_cast<int>(kl * dilation) - static_cast<int>(padding);

							if (inputPos < 0 || inputPos >= static_cast<int>(inputLength)) {
								continue;
							}

							const size_t inputIndex = (n * inputChannels + ic) * inputLength + static_cast<size_t>(inputPos);
							const size_t kernelIndex = (oc * inputChannels + ic) * kernelLength + kl;

							sum += input[inputIndex] * kernel[kernelIndex];
						}
					}

					if (bias) {
						sum += (*bias)[oc];
					}

					const size_t outputIndex = (n * outputChannels + oc) * outputLength + ol;

					output[outputIndex] = sum;
				}
			}
		}

		if (input.RequiresGrad() || kernel.RequiresGrad() || (bias && bias->RequiresGrad())) {
			output.SetRequiresGrad(true);
			output.SetGradFn(std::make_shared<AutoGrad::Conv1DGradFn<T>>(input.GetImpl(), kernel.GetImpl(), bias ? bias->GetImpl() : nullptr,
				stride, padding, dilation));
		}

		return output;
	}

	template <typename T>
	inline TensorCore::Tensor<T> Conv2D(const TensorCore::Tensor<T>& input, const TensorCore::Tensor<T>& kernel, const TensorCore::Tensor<T>* bias,
										size_t strideH, size_t strideW, 
										size_t paddingH, size_t paddingW, 
										size_t dilationH, size_t dilationW) {
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
			throw std::runtime_error("ERROR: Conv2D: Kernel must have 4 dimensions");
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
	
	template <typename T>
	inline TensorCore::Tensor<T> Conv3D(const TensorCore::Tensor<T>& input, const TensorCore::Tensor<T>& kernel, const TensorCore::Tensor<T>* bias,
										size_t strideD, size_t strideH, size_t strideW,
										size_t paddingD, size_t paddingH, size_t paddingW,
										size_t dilationD, size_t dilationH, size_t dilationW) {
		if (&input.GetAllocator() != &kernel.GetAllocator()) {
			throw std::runtime_error("ERROR: Operations between tensors on different allocators are forbidden");
		}

		if (bias && &input.GetAllocator() != &bias->GetAllocator()) {
			throw std::runtime_error("ERROR: Operations between tensors on different allocators are forbidden");
		}

		if (input.Rank() != 5) {
			throw std::runtime_error("ERROR: Conv3D: Input must have 4 dimensions");
		}

		if (kernel.Rank() != 5) {
			throw std::runtime_error("ERROR: Conv3D: Kernel must have 4 dimensions");
		}

		const size_t batchSize = input.Dims()[0];
		const size_t inputChannels = input.Dims()[1];
		const size_t inputDepth = input.Dims()[2];
		const size_t inputHeight = input.Dims()[3];
		const size_t inputWidth = input.Dims()[4];

		const size_t outputChannels = kernel.Dims()[0];
		const size_t kernelChannels = kernel.Dims()[1];
		const size_t kernelDepth = kernel.Dims()[2];
		const size_t kernelHeight = kernel.Dims()[3];
		const size_t kernelWidth = kernel.Dims()[4];

		if (inputChannels != kernelChannels) {
			throw std::runtime_error("ERROR: Conv3D: Input channels do not match kernel channels");
		}

		if (strideD == 0 || strideH == 0 || strideW == 0) {
			throw std::runtime_error("ERROR: Conv2D: Stride cannot be 0");
		}

		if (dilationD == 0 || dilationH == 0 || dilationW == 0) {
			throw std::runtime_error("ERROR: Conv2D: Dilation cannot be 0");
		}

		if (kernelDepth == 0 || kernelHeight == 0 || kernelWidth == 0) {
			throw std::runtime_error("ERROR: Conv2D: Kernel dimensions cannot be 0");
		}

		const size_t effectiveKernelDepth = dilationD * (kernelDepth - 1) + 1;
		const size_t effectiveKernelHeight = dilationH * (kernelHeight - 1) + 1;
		const size_t effectiveKernelWidth = dilationW * (kernelWidth - 1) + 1;

		const size_t paddedInputDepth = inputDepth + 2 * paddingD;
		const size_t paddedInputHeight = inputHeight + 2 * paddingH;
		const size_t paddedIputWidth = inputWidth + 2 * paddingW;

		if (paddedInputDepth < effectiveKernelDepth || paddedInputHeight < effectiveKernelHeight || paddedIputWidth < effectiveKernelWidth) {
			throw std::runtime_error("ERROR: Conv3D: Kernel is larger than padded input");
		}

		const size_t outputDepth = (paddedInputDepth - effectiveKernelDepth) / strideD + 1;
		const size_t outputHeight = (paddedInputHeight - effectiveKernelHeight) / strideH + 1;
		const size_t outputWidth = (paddedIputWidth - effectiveKernelWidth) / strideW + 1;

		Memory::ArenaAllocator& allocator = input.GetAllocator();

		TensorCore::Tensor<T> output{ {batchSize, outputChannels, outputDepth, outputHeight, outputWidth}, allocator };

		for (size_t n = 0; n < batchSize; ++n) {
			for (size_t oc = 0; oc < outputChannels; ++oc) {
				for (size_t od = 0; od < outputDepth; ++od) {
					for (size_t oh = 0; oh < outputHeight; ++oh) {
						for (size_t ow = 0; ow < outputWidth; ++ow) {
							T sum = static_cast<T>(0);

							for (size_t ic = 0; ic < inputChannels; ++ic) {
								for (size_t kd = 0; kd < kernelDepth; ++kd) {
									for (size_t kh = 0; kh < kernelHeight; ++kh) {
										for (size_t kw = 0; kw < kernelWidth; ++kw) {
											const int inputDepthPos = static_cast<int>(od * strideD) + static_cast<int>(kd * dilationD) - static_cast<int>(paddingD);
											const int inputRow = static_cast<int>(oh * strideH) + static_cast<int>(kh * dilationH) - static_cast<int>(paddingH);
											const int inputCol = static_cast<int>(ow * strideW) + static_cast<int>(kw * dilationW) - static_cast<int>(paddingW);

											if (inputDepthPos < 0 || inputDepthPos >= static_cast<int>(inputDepth) || inputRow < 0 || inputRow >= static_cast<int>(inputHeight) || inputCol < 0 || inputCol >= static_cast<int>(inputWidth)) {
												continue;
											}

											const size_t inputIndex = (((n * inputChannels + ic) * inputDepth + static_cast<size_t>(inputDepthPos)) * inputHeight + static_cast<size_t>(inputRow)) * inputWidth + static_cast<size_t>(inputCol);
											const size_t kernelIndex = (((oc * inputChannels + ic) * kernelDepth + kd) * kernelHeight + kh) * kernelWidth + kw;

											sum += input[inputIndex] * kernel[kernelIndex];
										}
									}
								}
							}

							if (bias) {
								sum += (*bias)[oc];
							}

							const size_t outputIndex = (((n * outputChannels + oc) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
							output[outputIndex] = sum;
						}
					}
				}
			}
		}

		if (input.RequiresGrad() || kernel.RequiresGrad() || (bias && bias->RequiresGrad())) {
			output.SetRequiresGrad(true);
			output.SetGradFn(std::make_shared<AutoGrad::Conv3DGradFn<T>>(input.GetImpl(), kernel.GetImpl(), bias ? bias->GetImpl() : nullptr,
				strideD, strideH, strideW, paddingD, paddingH, paddingW, dilationD, dilationH, dilationW));
		}

		return output;
	}
}