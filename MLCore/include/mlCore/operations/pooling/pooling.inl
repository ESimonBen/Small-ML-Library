/// pooling.inl
#include <numeric>
#include <optional>
#include <mlCore/autograd/functions/pooling/poolingGradFn.h>

namespace MLCore::Operations {
	inline size_t ComputePoolOutputSize(size_t inputSize, size_t filterSize, size_t stride, size_t padding, size_t dilation, bool ceilMode) {
		const size_t effectiveKernelSize = dilation * (filterSize - 1) + 1;
		const size_t paddedInputSize = inputSize + 2 * padding;

		if (paddedInputSize < effectiveKernelSize) {
			throw std::runtime_error("ERROR: Pool*D: Filter is larger than padded input");
		}

		if (ceilMode) {
			return ((paddedInputSize - effectiveKernelSize) + stride - 1) / stride + 1;
		}

		return (paddedInputSize - effectiveKernelSize) / stride + 1;
	}

	template <typename T>
	inline TensorCore::Tensor<T> MaxPool1D(const TensorCore::Tensor<T>& input, size_t filterLength, size_t stride, size_t padding, size_t dilation, bool ceilMode) {
		if (input.Rank() != 3) {
			throw std::runtime_error("ERROR: MaxPool1D: Input must have 3 dimensions");
		}

		if (stride == 0) {
			throw std::runtime_error("ERROR: MaxPool1D: Stride cannot be 0");
		}

		if (dilation == 0) {
			throw std::runtime_error("ERROR: MaxPool1D: Dilation cannot be 0");
		}

		if (filterLength == 0) {
			throw std::runtime_error("ERROR: MaxPool1D: Filter dimensions cannot be 0");
		}

		const size_t batchSize = input.Dims()[0];
		const size_t channels = input.Dims()[1];
		const size_t inputLength = input.Dims()[2];

		const size_t outputLength = ComputePoolOutputSize(inputLength, filterLength, stride, padding, dilation, ceilMode);

		TensorCore::Tensor<T> output{ {batchSize, channels, outputLength} };

		std::optional<TensorCore::Tensor<T>> indices;

		if (input.RequiresGrad()) {
			indices.emplace(TensorCore::Tensor<T>{{batchSize, channels, outputLength}});
		}

		if (input.IsContiguous()) {
			const T* inputData = input.Data();
			T* outputData = output.Data();
			T* indicesData = indices ? indices.value().Data() : nullptr;

			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t c = 0; c < channels; ++c) {
					for (size_t ol = 0; ol < outputLength; ++ol) {
						T max = std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
						size_t maxIdx = -1;

						for (size_t fl = 0; fl < filterLength; ++fl) {
							const int inputPos = static_cast<int>(ol * stride) + static_cast<int>(fl * dilation) - static_cast<int>(padding);

							if (inputPos < 0 || inputPos >= static_cast<int>(inputLength) ) {
								continue;
							}

							const size_t inputIndex = (n * channels + c) * inputLength + static_cast<size_t>(inputPos);
							T val = inputData[inputIndex];

							if (val > max) {
								max = val;
								maxIdx = static_cast<T>(inputIndex);
							}
						}

						const size_t outputIndex = (n * channels + c) * outputLength + ol;
						outputData[outputIndex] = max;
						if (indicesData) {
							indicesData[outputIndex] = maxIdx;
						}
					}
				}
			}
		}
		else {
			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t c = 0; c < channels; ++c) {
					for (size_t ol = 0; ol < outputLength; ++ol) {
						T max = std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
						size_t maxIdx = -1;

						for (size_t fl = 0; fl < filterLength; ++fl) {
							const int inputPos = static_cast<int>(ol * stride) + static_cast<int>(fl * dilation) - static_cast<int>(padding);

							if (inputPos < 0 || inputPos >= static_cast<int>(inputLength)) {
								continue;
							}

							const size_t inputIndex = (n * channels + c) * inputLength + static_cast<size_t>(inputPos);
							T val = input[inputIndex];

							if (val > max) {
								max = val;
								maxIdx = static_cast<T>(inputIndex);
							}
						}

						const size_t outputIndex = (n * channels + c) * outputLength + ol;
						output[outputIndex] = max;
						if (indices) {
							indices.value()[outputIndex] = maxIdx;
						}
					}
				}
			}
		}

		if (input.RequiresGrad()) {
			output.SetRequiresGrad(true);
			output.SetGradFn(std::make_shared<AutoGrad::MaxPool1DGradFn<T>>(input.GetImpl(), indices.value().GetImpl(), stride, padding, dilation));
		}

		return output;
	}

	template <typename T>
	inline TensorCore::Tensor<T> MaxPool2D(const TensorCore::Tensor<T>& input, size_t filterHeight, size_t filterWidth,
										   size_t strideH, size_t strideW, size_t paddingH, size_t paddingW, size_t dilationH, size_t dilationW, bool ceilMode) {
		if (input.Rank() != 4) {
			throw std::runtime_error("ERROR: MaxPool2D: Input must have 4 dimensions");
		}

		if (strideH == 0 || strideW == 0) {
			throw std::runtime_error("ERROR: MaxPool2D: Stride cannot be 0");
		}

		if (dilationH == 0 || dilationW == 0) {
			throw std::runtime_error("ERROR: MaxPool2D: Dilation cannot be 0");
		}

		if (filterHeight == 0 || filterWidth == 0) {
			throw std::runtime_error("ERROR: MaxPool2D: Filter dimensions cannot be 0");
		}

		const size_t batchSize = input.Dims()[0];
		const size_t channels = input.Dims()[1];
		const size_t inputHeight = input.Dims()[2];
		const size_t inputWidth = input.Dims()[3];

		const size_t outputHeight = ComputePoolOutputSize(inputHeight, filterHeight, strideH, paddingH, dilationH, ceilMode);
		const size_t outputWidth = ComputePoolOutputSize(inputWidth, filterWidth, strideW, paddingW, dilationW, ceilMode);

		TensorCore::Tensor<T> output{ {batchSize, channels, outputHeight, outputWidth} };

		std::optional<TensorCore::Tensor<T>> indices;
		
		if (input.RequiresGrad()) {
			indices.emplace(TensorCore::Tensor<T>{{batchSize, channels, outputHeight, outputWidth}});
		}

		if (input.IsContiguous()) {
			const T* inputData = input.Data();
			T* outputData = output.Data();
			T* indicesData = indices ? indices.value().Data() : nullptr;

			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t c = 0; c < channels; ++c) {
					for (size_t oh = 0; oh < outputHeight; ++oh) {
						for (size_t ow = 0; ow < outputWidth; ++ow) {
							T max = std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
							size_t maxIdx = -1;

							for (size_t fh = 0; fh < filterHeight; ++fh) {
								for (size_t fw = 0; fw < filterWidth; ++fw) {
									const int inputRow = static_cast<int>(oh * strideH) + static_cast<int>(fh * dilationH) - static_cast<int>(paddingH);
									const int inputCol = static_cast<int>(ow * strideW) + static_cast<int>(fw * dilationW) - static_cast<int>(paddingW);

									if (inputRow < 0 || inputRow >= static_cast<int>(inputHeight) || inputCol < 0 || inputCol >= static_cast<int>(inputWidth)) {
										continue;
									}

									const size_t inputIndex = ((n * channels + c) * inputHeight + static_cast<size_t>(inputRow)) * inputWidth + static_cast<size_t>(inputCol);
									T val = inputData[inputIndex];

									if (val > max) {
										max = val;
										maxIdx = static_cast<T>(inputIndex);
									}
								}
							}

							const size_t outputIndex = ((n * channels + c) * outputHeight + oh) * outputWidth + ow;
							outputData[outputIndex] = max;
							if (indicesData) {
								indicesData[outputIndex] = maxIdx;
							}
						}
					}
				}
			}
		}
		else {
			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t c = 0; c < channels; ++c) {
					for (size_t oh = 0; oh < outputHeight; ++oh) {
						for (size_t ow = 0; ow < outputWidth; ++ow) {
							T max = std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
							size_t maxIdx = -1;

							for (size_t fh = 0; fh < filterHeight; ++fh) {
								for (size_t fw = 0; fw < filterWidth; ++fw) {
									const int inputRow = static_cast<int>(oh * strideH) + static_cast<int>(fh * dilationH) - static_cast<int>(paddingH);
									const int inputCol = static_cast<int>(ow * strideW) + static_cast<int>(fw * dilationW) - static_cast<int>(paddingW);

									if (inputRow < 0 || inputRow >= static_cast<int>(inputHeight) || inputCol < 0 || inputCol >= static_cast<int>(inputWidth)) {
										continue;
									}

									const size_t inputIndex = ((n * channels + c) * inputHeight + static_cast<size_t>(inputRow)) * inputWidth + static_cast<size_t>(inputCol);
									T val = input[inputIndex];

									if (val > max) {
										max = val;
										maxIdx = static_cast<T>(inputIndex);
									}
								}
							}

							const size_t outputIndex = ((n * channels + c) * outputHeight + oh) * outputWidth + ow;
							output[outputIndex] = max;
							if (indices) {
								indices.value()[outputIndex] = maxIdx;
							}
						}
					}
				}
			}
		}

		if (input.RequiresGrad()) {
			output.SetRequiresGrad(true);
			output.SetGradFn(std::make_shared<AutoGrad::MaxPool2DGradFn<T>>(input.GetImpl(), indices.value().GetImpl(), strideH, strideW, paddingH, paddingW, dilationH, dilationW));
		}

		return output;
	}

	template <typename T>
	inline TensorCore::Tensor<T> MaxPool3D(const TensorCore::Tensor<T>& input, size_t filterDepth, size_t filterHeight, size_t filterWidth,
											size_t strideD, size_t strideH, size_t strideW,
											size_t paddingD, size_t paddingH, size_t paddingW,
											size_t dilationD, size_t dilationH, size_t dilationW, bool ceilMode) {
		if (input.Rank() != 5) {
			throw std::runtime_error("ERROR: MaxPool3D: Input must have 5 dimensions");
		}

		if (strideD == 0 || strideH == 0 || strideW == 0) {
			throw std::runtime_error("ERROR: MaxPool3D: Stride cannot be 0");
		}

		if (dilationD == 0 || dilationH == 0 || dilationW == 0) {
			throw std::runtime_error("ERROR: MaxPool3D: Dilation cannot be 0");
		}

		if (filterDepth == 0 || filterHeight == 0 || filterWidth == 0) {
			throw std::runtime_error("ERROR: MaxPool3D: Filter dimensions cannot be 0");
		}

		const size_t batchSize = input.Dims()[0];
		const size_t channels = input.Dims()[1];
		const size_t inputDepth = input.Dims()[2];
		const size_t inputHeight = input.Dims()[3];
		const size_t inputWidth = input.Dims()[4];

		const size_t outputDepth = ComputePoolOutputSize(inputDepth, filterDepth, strideD, paddingD, dilationD, ceilMode);
		const size_t outputHeight = ComputePoolOutputSize(inputHeight, filterHeight, strideH, paddingH, dilationH, ceilMode);
		const size_t outputWidth = ComputePoolOutputSize(inputWidth, filterWidth, strideW, paddingW, dilationW, ceilMode);

		TensorCore::Tensor<T> output{ {batchSize, channels, outputDepth, outputHeight, outputWidth} };

		std::optional<TensorCore::Tensor<T>> indices;

		if (input.RequiresGrad()) {
			indices.emplace(TensorCore::Tensor<T>{{batchSize, channels, outputDepth, outputHeight, outputWidth}});
		}

		if (input.IsContiguous()) {
			const T* inputData = input.Data();
			T* outputData = output.Data();
			T* indicesData = indices ? indices.value().Data() : nullptr;

			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t c = 0; c < channels; ++c) {
					for (size_t od = 0; od < outputDepth; ++od) {
						for (size_t oh = 0; oh < outputHeight; ++oh) {
							for (size_t ow = 0; ow < outputWidth; ++ow) {
								T max = std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
								size_t maxIdx = -1;

								for (size_t fd = 0; fd < filterDepth; ++fd) {
									for (size_t fh = 0; fh < filterHeight; ++fh) {
										for (size_t fw = 0; fw < filterWidth; ++fw) {
											const int inputDepthPos = static_cast<int>(od * strideD) + static_cast<int>(fd * dilationD) - static_cast<int>(paddingD);
											const int inputRow = static_cast<int>(oh * strideH) + static_cast<int>(fh * dilationH) - static_cast<int>(paddingH);
											const int inputCol = static_cast<int>(ow * strideW) + static_cast<int>(fw * dilationW) - static_cast<int>(paddingW);

											if (inputDepthPos < 0 || inputDepthPos >= static_cast<int>(inputDepth) || inputRow < 0 || inputRow >= static_cast<int>(inputHeight) || inputCol < 0 || inputCol >= static_cast<int>(inputWidth)) {
												continue;
											}

											const size_t inputIndex = (((n * channels + c) * inputDepth + static_cast<size_t>(inputDepthPos)) * inputHeight + static_cast<size_t>(inputRow)) * inputWidth + static_cast<size_t>(inputCol);
											T val = inputData[inputIndex];

											if (val > max) {
												max = val;
												maxIdx = static_cast<T>(inputIndex);
											}
										}
									}
								}

								const size_t outputIndex = (((n * channels + c) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
								outputData[outputIndex] = max;
								if (indicesData) {
									indicesData[outputIndex] = maxIdx;
								}
							}
						}
					}
				}
			}
		}
		else {
			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t c = 0; c < channels; ++c) {
					for (size_t od = 0; od < outputDepth; ++od) {
						for (size_t oh = 0; oh < outputHeight; ++oh) {
							for (size_t ow = 0; ow < outputWidth; ++ow) {
								T max = std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
								size_t maxIdx = -1;

								for (size_t fd = 0; fd < filterDepth; ++fd) {
									for (size_t fh = 0; fh < filterHeight; ++fh) {
										for (size_t fw = 0; fw < filterWidth; ++fw) {
											const int inputDepthPos = static_cast<int>(od * strideD) + static_cast<int>(fd * dilationD) - static_cast<int>(paddingD);
											const int inputRow = static_cast<int>(oh * strideH) + static_cast<int>(fh * dilationH) - static_cast<int>(paddingH);
											const int inputCol = static_cast<int>(ow * strideW) + static_cast<int>(fw * dilationW) - static_cast<int>(paddingW);

											if (inputDepthPos < 0 || inputDepthPos >= static_cast<int>(inputDepth) || inputRow < 0 || inputRow >= static_cast<int>(inputHeight) || inputCol < 0 || inputCol >= static_cast<int>(inputWidth)) {
												continue;
											}

											const size_t inputIndex = (((n * channels + c) * inputDepth + static_cast<size_t>(inputDepthPos)) * inputHeight + static_cast<size_t>(inputRow)) * inputWidth + static_cast<size_t>(inputCol);
											T val = input[inputIndex];

											if (val > max) {
												max = val;
												maxIdx = static_cast<T>(inputIndex);
											}
										}
									}
								}

								const size_t outputIndex = ((n * channels + c) * outputHeight + oh) * outputWidth + ow;
								output[outputIndex] = max;
								if (indices) {
									indices.value()[outputIndex] = maxIdx;
								}
							}
						}
					}
				}
			}
		}

		if (input.RequiresGrad()) {
			output.SetRequiresGrad(true);
			output.SetGradFn(std::make_shared<AutoGrad::MaxPool3DGradFn<T>>(input.GetImpl(), indices.value().GetImpl(), strideD, strideH, strideW, paddingD, 
																			paddingH, paddingW, dilationD, dilationH, dilationW));
		}

		return output;
	}

	template <typename T>
	inline TensorCore::Tensor<T> MaxPool1D(const TensorCore::Tensor<T>& input, size_t filterLength, size_t padding, size_t dilation, bool ceilMode) {
		return MaxPool1D(input, filterLength, filterLength, padding, dilation, ceilMode);
	}

	template <typename T>
	inline TensorCore::Tensor<T> MaxPool2D(const TensorCore::Tensor<T>& input, size_t filterHeight, size_t filterWidth, size_t paddingH, size_t paddingW,
										   size_t dilationH, size_t dilationW, bool ceilMode) {
		return MaxPool2D(input, filterHeight, filterWidth, filterHeight, filterWidth, paddingH, paddingW, dilationH, dilationW, ceilMode);
	}

	template <typename T>
	inline TensorCore::Tensor<T> MaxPool3D(const TensorCore::Tensor<T>& input, size_t filterDepth, size_t filterHeight, size_t filterWidth,
										   size_t paddingD, size_t paddingH, size_t paddingW,
										   size_t dilationD, size_t dilationH, size_t dilationW, bool ceilMode) {
		return MaxPool3D(input, filterDepth, filterHeight, filterWidth, filterDepth, filterHeight, filterWidth, 
						 paddingD, paddingH, paddingW, dilationD, dilationH, dilationW, ceilMode);
	}
}