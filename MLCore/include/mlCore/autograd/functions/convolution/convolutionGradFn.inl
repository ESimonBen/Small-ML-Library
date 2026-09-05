/// convolutionGradFn.inl
#include <optional>
#include <mlCore/operations/convolution/convolution.h>

namespace MLCore::AutoGrad {
	template <typename T>
	inline Conv1DGradFn<T>::Conv1DGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> input, std::shared_ptr<TensorCore::TensorImpl<T>> kernel, std::shared_ptr<TensorCore::TensorImpl<T>> bias,
										 size_t stride, size_t padding, size_t dilation)
		: GradFn<T>({input, kernel, bias}), m_Stride(stride), m_Padding(padding), m_Dilation(dilation)
	{}
	
	template <typename T>
	inline void Conv1DGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (!this->inputs[0]) {
			throw std::runtime_error("ERROR: Conv1DGradFn: Null input");
		}

		if (!this->inputs[1]) {
			throw std::runtime_error("ERROR: Conv1DGradFn: Null kernel");
		}

		TensorCore::Tensor<T> input{ this->inputs[0] };
		TensorCore::Tensor<T> kernel{ this->inputs[1] };

		std::optional<TensorCore::Tensor<T>> bias;

		if (this->inputs.size() > 2 && this->inputs[2]) {
			bias.emplace(TensorCore::Tensor<T>{this->inputs[2]});
		}

		const bool requiresInputGrad = input.RequiresGrad();
		const bool requiresKernelGrad = kernel.RequiresGrad();
		const bool requiresBiasGrad = bias ? bias.value().RequiresGrad() : false;

		if (!requiresInputGrad && !requiresKernelGrad && !requiresBiasGrad) {
			return;
		}

		if (gradOutput.Rank() != 3) {
			throw std::runtime_error("ERROR: Conv1DGradFn: Output gradient must have 3 dimensions");
		}

		const size_t batchSize = input.Dims()[0];
		const size_t inputChannels = input.Dims()[1];
		const size_t inputLength = input.Dims()[2];

		const size_t outputChannels = kernel.Dims()[0];
		const size_t kernelChannels = kernel.Dims()[1];
		const size_t kernelLength = kernel.Dims()[2];

		const size_t expectedOutputLength = Operations::ComputeConvOutputSize(inputLength, kernelLength, m_Stride, m_Padding, m_Dilation);

		if (gradOutput.Dims()[2] != expectedOutputLength) {
			throw std::runtime_error("ERROR: Conv1DGradFn: Output gradient length mismatch");
		}

		const size_t outputLength = gradOutput.Dims()[2];

		if (inputChannels != kernelChannels) {
			throw std::runtime_error("ERROR: Conv1DGradFn: Input channels do not match kernel channels");
		}

		if (gradOutput.Dims()[0] != batchSize) {
			throw std::runtime_error("ERROR: Conv1DGradFn: Output gradient batch size mismatch");
		}

		if (gradOutput.Dims()[1] != outputChannels) {
			throw std::runtime_error("ERROR: Conv1DGradFn: Output gradient channel count mismatch");
		}

		Memory::ArenaAllocator& allocator = input.GetAllocator();

		std::optional<TensorCore::Tensor<T>> gradInput;
		std::optional<TensorCore::Tensor<T>> gradKernel;
		std::optional<TensorCore::Tensor<T>> gradBias;

		if (requiresInputGrad) {
			gradInput.emplace(TensorCore::Tensor<T>{{batchSize, inputChannels, inputLength}, allocator});
			gradInput.value().Fill(static_cast<T>(0));
		}

		if (requiresKernelGrad) {
			gradKernel.emplace(TensorCore::Tensor<T>{{outputChannels, inputChannels, kernelLength}, allocator});
			gradKernel.value().Fill(static_cast<T>(0));
		}

		if (requiresBiasGrad) {
			gradBias.emplace(TensorCore::Tensor<T>{{outputChannels}, allocator});
			gradBias.value().Fill(static_cast<T>(0));
		}

		if (gradOutput.IsContiguous() && input.IsContiguous() && kernel.IsContiguous() && (bias && bias.value().IsContiguous())) {
			const T* gradOutputData = gradOutput.Data();
			const T* inputData = input.Data();
			const T* kernelData = kernel.Data();
			T* gradInputData = gradInput ? gradInput.value().Data() : nullptr;
			T* gradKernelData = gradKernel ? gradKernel.value().Data() : nullptr;
			T* gradBiasData = gradBias ? gradBias.value().Data() : nullptr;

			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t oc = 0; oc < outputChannels; ++oc) {
					for (size_t ol = 0; ol < outputLength; ++ol) {
						size_t gradIndex = (n * outputChannels + oc) * outputLength + ol;

						const T grad = gradOutputData[gradIndex];

						if (requiresBiasGrad) {
							gradBiasData[oc] += grad;
						}

						for (size_t ic = 0; ic < inputChannels; ++ic) {
							for (size_t kl = 0; kl < kernelLength; ++kl) {
								const size_t kernelOffset = kl * m_Dilation;

								const int inputPos = static_cast<int>(ol * m_Stride) + static_cast<int>(kernelOffset) - static_cast<int>(m_Padding);

								if (inputPos < 0 || inputPos >= static_cast<int>(inputLength)) {
									continue;
								}

								const size_t inputIndex = (n * inputChannels + ic) * inputLength + static_cast<size_t>(inputPos);
								const size_t kernelIndex = (oc * inputChannels + ic) * kernelLength + kl;

								if (requiresInputGrad) {
									gradInputData[inputIndex] += grad * kernelData[kernelIndex];
								}

								if (requiresKernelGrad) {
									gradKernelData[kernelIndex] += grad * inputData[inputIndex];
								}
							}
						}
					}
				}
			}
		}
		else {
			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t oc = 0; oc < outputChannels; ++oc) {
					for (size_t ol = 0; ol < outputLength; ++ol) {
						size_t gradIndex = (n * outputChannels + oc) * outputLength + ol;

						const T grad = gradOutput[gradIndex];

						if (requiresBiasGrad) {
							auto& gb = gradBias.value();
							gradBias.value()[oc] += grad;
						}

						for (size_t ic = 0; ic < inputChannels; ++ic) {
							for (size_t kl = 0; kl < kernelLength; ++kl) {
								const size_t kernelOffset = kl * m_Dilation;

								const int inputPos = static_cast<int>(ol * m_Stride) + static_cast<int>(kernelOffset) - static_cast<int>(m_Padding);

								if (inputPos < 0 || inputPos >= static_cast<int>(inputLength)) {
									continue;
								}

								const size_t inputIndex = (n * inputChannels + ic) * inputLength + static_cast<size_t>(inputPos);
								const size_t kernelIndex = (oc * inputChannels + ic) * kernelLength + kl;

								if (requiresInputGrad) {
									auto& gi = gradInput.value();
									gi[inputIndex] += grad * kernel[kernelIndex];
								}

								if (requiresKernelGrad) {
									auto& gk = gradKernel.value();
									gk[kernelIndex] += grad * input[inputIndex];
								}
							}
						}
					}
				}
			}
		}

		if (requiresInputGrad) {
			input.Backward(gradInput.value());
		}

		if (requiresKernelGrad) {
			kernel.Backward(gradKernel.value());
		}

		if (requiresBiasGrad) {
			bias.value().Backward(gradBias.value());
		}
	}

	template <typename T>
	inline Conv2DGradFn<T>::Conv2DGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> input, std::shared_ptr<TensorCore::TensorImpl<T>> kernel, std::shared_ptr<TensorCore::TensorImpl<T>> bias,
										 size_t strideH, size_t strideW,
										 size_t paddingH, size_t paddingW,
										 size_t dilationH, size_t dilationW)
		: GradFn<T>({ input, kernel, bias }), m_StrideH(strideH), m_StrideW(strideW), m_PaddingH(paddingH), m_PaddingW(paddingW), m_DilationH(dilationH), m_DilationW(dilationW)
	{}
	
	template <typename T>
	inline void Conv2DGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (!this->inputs[0]) {
			throw std::runtime_error("ERROR: Conv2DGradFn: Null input");
		}

		if (!this->inputs[1]) {
			throw std::runtime_error("ERROR: Conv2DGradFn: Null kernel");
		}
		
		TensorCore::Tensor<T> input{ this->inputs[0] };
		TensorCore::Tensor<T> kernel{ this->inputs[1] };

		std::optional<TensorCore::Tensor<T>> bias;
		
		if (this->inputs.size() > 2 && this->inputs[2]) {
			bias.emplace(TensorCore::Tensor<T>{this->inputs[2]});
		}

		const bool requiresInputGrad = input.RequiresGrad();
		const bool requiresKernelGrad = kernel.RequiresGrad();
		const bool requiresBiasGrad = bias ? bias.value().RequiresGrad() : false;

		if (!requiresInputGrad && !requiresKernelGrad && !requiresBiasGrad) {
			return;
		}

		if (gradOutput.Rank() != 4) {
			throw std::runtime_error("ERROR: Conv2DGradFn: Output gradient must have 4 dimensions");
		}

		const size_t batchSize = input.Dims()[0];
		const size_t inputChannels = input.Dims()[1];
		const size_t inputHeight = input.Dims()[2];
		const size_t inputWidth = input.Dims()[3];

		const size_t outputChannels = kernel.Dims()[0];
		const size_t kernelChannels = kernel.Dims()[1];
		const size_t kernelHeight = kernel.Dims()[2];
		const size_t kernelWidth = kernel.Dims()[3];

		const size_t expectedOutputHeight = Operations::ComputeConvOutputSize(inputHeight, kernelHeight, m_StrideH, m_PaddingH, m_DilationH);
		const size_t expectedOutputWidth = Operations::ComputeConvOutputSize(inputWidth, kernelWidth, m_StrideW, m_PaddingW, m_DilationW);

		if (gradOutput.Dims()[2] != expectedOutputHeight || gradOutput.Dims()[3] != expectedOutputWidth) {
			throw std::runtime_error("ERROR: Conv2DGradFn: Output gradient spatial dimensions mismatch");
		}

		const size_t outputHeight = gradOutput.Dims()[2];
		const size_t outputWidth = gradOutput.Dims()[3];

		if (inputChannels != kernelChannels) {
			throw std::runtime_error("ERROR: Conv2DGradFn: Input channels do not match kernel channels");
		}

		if (gradOutput.Dims()[0] != batchSize) {
			throw std::runtime_error("ERROR: Conv2DGradFn: Output gradient batch size mismatch");
		}

		if (gradOutput.Dims()[1] != outputChannels) {
			throw std::runtime_error("ERROR: Conv2DGradFn: Output gradient channel count mismatch");
		}

		Memory::ArenaAllocator& allocator = input.GetAllocator();

		std::optional<TensorCore::Tensor<T>> gradInput;
		std::optional<TensorCore::Tensor<T>> gradKernel;
		std::optional<TensorCore::Tensor<T>> gradBias;

		if (requiresInputGrad) {
			gradInput.emplace(TensorCore::Tensor<T>{{batchSize, inputChannels, inputHeight, inputWidth}, allocator});
			gradInput.value().Fill(static_cast<T>(0));
		}

		if (requiresKernelGrad) {
			gradKernel.emplace(TensorCore::Tensor<T>{{outputChannels, inputChannels, kernelHeight, kernelWidth}, allocator});
			gradKernel.value().Fill(static_cast<T>(0));
		}

		if (requiresBiasGrad) {
			gradBias.emplace(TensorCore::Tensor<T>{{outputChannels}, allocator});
			gradBias.value().Fill(static_cast<T>(0));
		}

		if (gradOutput.IsContiguous() && input.IsContiguous() && kernel.IsContiguous() && (bias && bias.value().IsContiguous())) {
			const T* gradOutputData = gradOutput.Data();
			const T* inputData = input.Data();
			const T* kernelData = kernel.Data();
			T* gradInputData = gradInput ? gradInput.value().Data() : nullptr;
			T* gradKernelData = gradKernel ? gradKernel.value().Data() : nullptr;
			T* gradBiasData = gradBias ? gradBias.value().Data() : nullptr;

			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t oc = 0; oc < outputChannels; ++oc) {
					for (size_t oh = 0; oh < outputHeight; ++oh) {
						for (size_t ow = 0; ow < outputWidth; ++ow) {
							size_t gradIndex = ((n * outputChannels + oc) * outputHeight + oh) * outputWidth + ow;

							const T grad = gradOutputData[gradIndex];

							if (requiresBiasGrad) {
								gradBiasData[oc] += grad;
							}

							for (size_t ic = 0; ic < inputChannels; ++ic) {
								for (size_t kh = 0; kh < kernelHeight; ++kh) {
									for (size_t kw = 0; kw < kernelWidth; ++kw) {
										const size_t kernelOffsetH = kh * m_DilationH;
										const size_t kernelOffsetW = kw * m_DilationW;

										const int inputRow = static_cast<int>(oh * m_StrideH) + static_cast<int>(kernelOffsetH) - static_cast<int>(m_PaddingH);
										const int inputCol = static_cast<int>(ow * m_StrideW) + static_cast<int>(kernelOffsetW) - static_cast<int>(m_PaddingW);

										if (inputRow < 0 || inputRow >= static_cast<int>(inputHeight) || inputCol < 0 || inputCol >= static_cast<int>(inputWidth)) {
											continue;
										}

										const size_t inputIndex = ((n * inputChannels + ic) * inputHeight + static_cast<size_t>(inputRow)) * inputWidth + static_cast<size_t>(inputCol);
										const size_t kernelIndex = ((oc * inputChannels + ic) * kernelHeight + kh) * kernelWidth + kw;

										if (requiresInputGrad) {
											gradInputData[inputIndex] += grad * kernelData[kernelIndex];
										}

										if (requiresKernelGrad) {
											gradKernelData[kernelIndex] += grad * inputData[inputIndex];
										}
									}
								}
							}
						}
					}
				}
			}
		}
		else {
			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t oc = 0; oc < outputChannels; ++oc) {
					for (size_t oh = 0; oh < outputHeight; ++oh) {
						for (size_t ow = 0; ow < outputWidth; ++ow) {
							size_t gradIndex = ((n * outputChannels + oc) * outputHeight + oh) * outputWidth + ow;

							const T grad = gradOutput[gradIndex];

							if (requiresBiasGrad) {
								auto& gb = gradBias.value();
								gb[oc] += grad;
							}

							for (size_t ic = 0; ic < inputChannels; ++ic) {
								for (size_t kh = 0; kh < kernelHeight; ++kh) {
									for (size_t kw = 0; kw < kernelWidth; ++kw) {
										const size_t kernelOffsetH = kh * m_DilationH;
										const size_t kernelOffsetW = kw * m_DilationW;

										const int inputRow = static_cast<int>(oh * m_StrideH) + static_cast<int>(kernelOffsetH) - static_cast<int>(m_PaddingH);
										const int inputCol = static_cast<int>(ow * m_StrideW) + static_cast<int>(kernelOffsetW) - static_cast<int>(m_PaddingW);

										if (inputRow < 0 || inputRow >= static_cast<int>(inputHeight) || inputCol < 0 || inputCol >= static_cast<int>(inputWidth)) {
											continue;
										}

										const size_t inputIndex = ((n * inputChannels + ic) * inputHeight + static_cast<size_t>(inputRow)) * inputWidth + static_cast<size_t>(inputCol);
										const size_t kernelIndex = ((oc * inputChannels + ic) * kernelHeight + kh) * kernelWidth + kw;

										if (requiresInputGrad) {
											auto& gi = gradInput.value();
											gi[inputIndex] += grad * kernel[kernelIndex];
										}

										if (requiresKernelGrad) {
											auto& gk = gradKernel.value();
											gk[kernelIndex] += grad * input[inputIndex];
										}
									}
								}
							}
						}
					}
				}
			}
		}

		if (requiresInputGrad) {
			input.Backward(gradInput.value());
		}

		if (requiresKernelGrad) {
			kernel.Backward(gradKernel.value());
		}

		if (requiresBiasGrad) {
			bias.value().Backward(gradBias.value());
		}
	}
	
	template <typename T>
	inline Conv3DGradFn<T>::Conv3DGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> input, std::shared_ptr<TensorCore::TensorImpl<T>> kernel, std::shared_ptr<TensorCore::TensorImpl<T>> bias,
										 size_t strideD, size_t strideH, size_t strideW,
										 size_t paddingD, size_t paddingH, size_t paddingW,
										 size_t dilationD, size_t dilationH, size_t dilationW)
		: GradFn<T>({input, kernel, bias}), m_StrideD(strideD), m_StrideH(strideH), m_StrideW(strideW), m_PaddingD(paddingD), m_PaddingH(paddingH), m_PaddingW(paddingW), m_DilationD(dilationD), m_DilationH(dilationH), m_DilationW(dilationW)
	{}
	
	template <typename T>
	inline void Conv3DGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (!this->inputs[0]) {
			throw std::runtime_error("ERROR: Conv3DGradFn: Null input");
		}

		if (!this->inputs[1]) {
			throw std::runtime_error("ERROR: Conv3DGradFn: Null kernel");
		}

		TensorCore::Tensor<T> input{ this->inputs[0] };
		TensorCore::Tensor<T> kernel{ this->inputs[1] };

		std::optional<TensorCore::Tensor<T>> bias;

		if (this->inputs.size() > 2 && this->inputs[2]) {
			bias.emplace(TensorCore::Tensor<T>{this->inputs[2]});
		}

		const bool requiresInputGrad = input.RequiresGrad();
		const bool requiresKernelGrad = kernel.RequiresGrad();
		const bool requiresBiasGrad = bias ? bias.value().RequiresGrad() : false;

		if (!requiresInputGrad && !requiresKernelGrad && !requiresBiasGrad) {
			return;
		}

		if (gradOutput.Rank() != 5) {
			throw std::runtime_error("ERROR: Conv3DGradFn: Output gradient must have 5 dimensions");
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

		const size_t expectedOutputDepth = Operations::ComputeConvOutputSize(inputDepth, kernelDepth, m_StrideD, m_PaddingD, m_DilationD);
		const size_t expectedOutputHeight = Operations::ComputeConvOutputSize(inputHeight, kernelHeight, m_StrideH, m_PaddingH, m_DilationH);
		const size_t expectedOutputWidth = Operations::ComputeConvOutputSize(inputWidth, kernelWidth, m_StrideW, m_PaddingW, m_DilationW);

		if (gradOutput.Dims()[2] != expectedOutputDepth || gradOutput.Dims()[3] != expectedOutputHeight || gradOutput.Dims()[4] != expectedOutputWidth) {
			throw std::runtime_error("ERROR: Conv3DGradFn: Output gradient spatial dimensions mismatch");
		}

		const size_t outputDepth = gradOutput.Dims()[2];
		const size_t outputHeight = gradOutput.Dims()[3];
		const size_t outputWidth = gradOutput.Dims()[4];

		if (inputChannels != kernelChannels) {
			throw std::runtime_error("ERROR: Conv3DGradFn: Input channels do not match kernel channels");
		}

		if (gradOutput.Dims()[0] != batchSize) {
			throw std::runtime_error("ERROR: Conv3DGradFn: Output gradient batch size mismatch");
		}

		if (gradOutput.Dims()[1] != outputChannels) {
			throw std::runtime_error("ERROR: Conv3DGradFn: Output gradient channel count mismatch");
		}

		Memory::ArenaAllocator& allocator = input.GetAllocator();

		std::optional<TensorCore::Tensor<T>> gradInput;
		std::optional<TensorCore::Tensor<T>> gradKernel;
		std::optional<TensorCore::Tensor<T>> gradBias;

		if (requiresInputGrad) {
			gradInput.emplace(TensorCore::Tensor<T>{{batchSize, inputChannels, inputDepth, inputHeight, inputWidth}, allocator});
			gradInput.value().Fill(static_cast<T>(0));
		}

		if (requiresKernelGrad) {
			gradKernel.emplace(TensorCore::Tensor<T>{{outputChannels, inputChannels, kernelDepth, kernelHeight, kernelWidth}, allocator});
			gradKernel.value().Fill(static_cast<T>(0));
		}

		if (requiresBiasGrad) {
			gradBias.emplace(TensorCore::Tensor<T>{{outputChannels}, allocator});
			gradBias.value().Fill(static_cast<T>(0));
		}

		if (gradOutput.IsContiguous() && input.IsContiguous() && kernel.IsContiguous() && (bias && bias.value().IsContiguous())) {
			const T* gradOutputData = gradOutput.Data();
			const T* inputData = input.Data();
			const T* kernelData = kernel.Data();
			T* gradInputData = gradInput ? gradInput.value().Data() : nullptr;
			T* gradKernelData = gradKernel ? gradKernel.value().Data() : nullptr;
			T* gradBiasData = gradBias ? gradBias.value().Data() : nullptr;

			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t oc = 0; oc < outputChannels; ++oc) {
					for (size_t od = 0; od < outputDepth; ++od) {
						for (size_t oh = 0; oh < outputHeight; ++oh) {
							for (size_t ow = 0; ow < outputWidth; ++ow) {
								size_t gradIndex = (((n * outputChannels + oc) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;

								const T grad = gradOutputData[gradIndex];

								if (requiresBiasGrad) {
									gradBiasData[oc] += grad;
								}

								for (size_t ic = 0; ic < inputChannels; ++ic) {
									for (size_t kd = 0; kd < kernelDepth; ++kd) {
										for (size_t kh = 0; kh < kernelHeight; ++kh) {
											for (size_t kw = 0; kw < kernelWidth; ++kw) {
												const size_t kernelOffsetD = kd * m_DilationD;
												const size_t kernelOffsetH = kh * m_DilationH;
												const size_t kernelOffsetW = kw * m_DilationW;

												const int inputDepthPos = static_cast<int>(od * m_StrideD) + static_cast<int>(kernelOffsetD) - static_cast<int>(m_PaddingD);
												const int inputRow = static_cast<int>(oh * m_StrideH) + static_cast<int>(kernelOffsetH) - static_cast<int>(m_PaddingH);
												const int inputCol = static_cast<int>(ow * m_StrideW) + static_cast<int>(kernelOffsetW) - static_cast<int>(m_PaddingW);

												if (inputDepthPos < 0 || inputDepthPos >= static_cast<int>(inputDepth) || inputRow < 0 || inputRow >= static_cast<int>(inputHeight) || inputCol < 0 || inputCol >= static_cast<int>(inputWidth)) {
													continue;
												}

												const size_t inputIndex = (((n * inputChannels + ic) * inputDepth + static_cast<size_t>(inputDepthPos)) * inputHeight + static_cast<size_t>(inputRow)) * inputWidth + static_cast<size_t>(inputCol);
												const size_t kernelIndex = (((oc * inputChannels + ic) * kernelDepth + kd) * kernelHeight + kh) * kernelWidth + kw;

												if (requiresInputGrad) {
													gradInputData[inputIndex] += grad * kernelData[kernelIndex];
												}

												if (requiresKernelGrad) {
													gradKernelData[kernelIndex] += grad * inputData[inputIndex];
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		else {
			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t oc = 0; oc < outputChannels; ++oc) {
					for (size_t od = 0; od < outputDepth; ++od) {
						for (size_t oh = 0; oh < outputHeight; ++oh) {
							for (size_t ow = 0; ow < outputWidth; ++ow) {
								size_t gradIndex = (((n * outputChannels + oc) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;

								const T grad = gradOutput[gradIndex];

								if (requiresBiasGrad) {
									auto& gb = gradBias.value();
									gb[oc] += grad;
								}

								for (size_t ic = 0; ic < inputChannels; ++ic) {
									for (size_t kd = 0; kd < kernelDepth; ++kd) {
										for (size_t kh = 0; kh < kernelHeight; ++kh) {
											for (size_t kw = 0; kw < kernelWidth; ++kw) {
												const size_t kernelOffsetD = kd * m_DilationD;
												const size_t kernelOffsetH = kh * m_DilationH;
												const size_t kernelOffsetW = kw * m_DilationW;

												const int inputDepthPos = static_cast<int>(od * m_StrideD) + static_cast<int>(kernelOffsetD) - static_cast<int>(m_PaddingD);
												const int inputRow = static_cast<int>(oh * m_StrideH) + static_cast<int>(kernelOffsetH) - static_cast<int>(m_PaddingH);
												const int inputCol = static_cast<int>(ow * m_StrideW) + static_cast<int>(kernelOffsetW) - static_cast<int>(m_PaddingW);

												if (inputDepthPos < 0 || inputDepthPos >= static_cast<int>(inputDepth) || inputRow < 0 || inputRow >= static_cast<int>(inputHeight) || inputCol < 0 || inputCol >= static_cast<int>(inputWidth)) {
													continue;
												}

												const size_t inputIndex = (((n * inputChannels + ic) * inputDepth + static_cast<size_t>(inputDepthPos)) * inputHeight + static_cast<size_t>(inputRow)) * inputWidth + static_cast<size_t>(inputCol);
												const size_t kernelIndex = (((oc * inputChannels + ic) * kernelDepth + kd) * kernelHeight + kh) * kernelWidth + kw;

												if (requiresInputGrad) {
													auto& gi = gradInput.value();
													gi[inputIndex] += grad * kernel[kernelIndex];
												}

												if (requiresKernelGrad) {
													auto& gk = gradKernel.value();
													gk[kernelIndex] += grad * input[inputIndex];
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

		if (requiresInputGrad) {
			input.Backward(gradInput.value());
		}

		if (requiresKernelGrad) {
			kernel.Backward(gradKernel.value());
		}

		if (requiresBiasGrad) {
			bias.value().Backward(gradBias.value());
		}
	}
}