/// poolingGradFn.inl

namespace MLCore::AutoGrad {
	template <typename T>
	MaxPool1DGradFn<T>::MaxPool1DGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> input, std::shared_ptr<TensorCore::TensorImpl<T>> indices, size_t stride, size_t padding, size_t dilation)
		: GradFn<T>({input, indices}), m_Stride(stride), m_Padding(padding), m_Dilation(dilation)
	{}

	template <typename T>
	void MaxPool1DGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (!this->inputs[0]) {
			throw std::runtime_error("ERROR: MaxPool1DGradFn: Null input");
		}

		if (!this->inputs[0]) {
			throw std::runtime_error("ERROR: MaxPool1DGradFn: Null indices");
		}

		TensorCore::Tensor<T> input{ this->inputs[0] };
		TensorCore::Tensor<T> indices{ this->inputs[1] };

		if (!input.RequiresGrad()) {
			return;
		}

		if (gradOutput.Rank() != 3) {
			throw std::runtime_error("ERROR: MaxPool1DGradFn: Output gradient must have 3 dimensions");
		}

		const size_t batchSize = input.Dims()[0];
		const size_t channels = input.Dims()[1];
		const size_t inputLength = input.Dims()[2];

		if (gradOutput.Dims()[2] != indices.Dims()[2]) {
			throw std::runtime_error("ERROR: MaxPool1DGradFn: Output gradient spatial dimensions mismatch");
		}

		const size_t outputLength = gradOutput.Dims()[2];

		if (gradOutput.Dims()[0] != batchSize) {
			throw std::runtime_error("ERROR: MaxPool1DGradFn: Output gradient batch size mismatch");
		}

		if (gradOutput.Dims()[1] != channels) {
			throw std::runtime_error("ERROR: MaxPool1DGradFn: Output gradient channel size mismatch");
		}

		Memory::ArenaAllocator& allocator = input.GetAllocator();

		TensorCore::Tensor<T> gradInput{ {batchSize, channels, inputLength}, allocator };
		gradInput.Fill(static_cast<T>(0));

		if (input.IsContiguous() && indices.IsContiguous()) {
			const T* gradOutputData = gradOutput.Data();
			const T* indicesData = indices.Data();
			T* gradInputData = gradInput.Data();

			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t c = 0; c < channels; ++c) {
					for (size_t ol = 0; ol < outputLength; ++ol) {
						size_t gradIndex = (n * channels + c) * outputLength + ol;

						T grad = gradOutputData[gradIndex];
						T maxIdx = indicesData[gradIndex];

						if (maxIdx >= 0) {
							gradInputData[static_cast<size_t>(maxIdx)] += grad;
						}
					}
				}
			}
		}
		else {
			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t c = 0; c < channels; ++c) {
					for (size_t ol = 0; ol < outputLength; ++ol) {
						size_t gradIndex = (n * channels + c) * outputLength + ol;

						T grad = gradOutput[gradIndex];
						T maxIdx = indices[gradIndex];

						if (maxIdx >= 0) {
							gradInput[maxIdx] += grad;
						}
					}
				}
			}
		}

		input.Backward(gradInput);
	}

	template <typename T>
	MaxPool2DGradFn<T>::MaxPool2DGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> input, std::shared_ptr<TensorCore::TensorImpl<T>> indices,
										size_t strideH, size_t strideW, size_t paddingH, size_t paddingW, size_t dilationH, size_t dilationW)
		: GradFn<T>({input, indices}), m_StrideH(strideH), m_StrideW(strideW), m_PaddingH(paddingH), m_PaddingW(paddingW), m_DilationH(dilationH), m_DilationW(dilationW)
	{}

	template <typename T>
	void MaxPool2DGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (!this->inputs[0]) {
			throw std::runtime_error("ERROR: MaxPool2DGradFn: Null input");
		}

		if (!this->inputs[0]) {
			throw std::runtime_error("ERROR: MaxPool2DGradFn: Null indices");
		}

		TensorCore::Tensor<T> input{ this->inputs[0] };
		TensorCore::Tensor<T> indices{ this->inputs[1] };

		if (!input.RequiresGrad()) {
			return;
		}

		if (gradOutput.Rank() != 4) {
			throw std::runtime_error("ERROR: MaxPool2DGradFn: Output gradient must have 4 dimensions");
		}

		const size_t batchSize = input.Dims()[0];
		const size_t channels = input.Dims()[1];
		const size_t inputHeight = input.Dims()[2];
		const size_t inputWidth = input.Dims()[3];

		if (gradOutput.Dims()[2] != indices.Dims()[2] || gradOutput.Dims()[3] != indices.Dims()[3]) {
			throw std::runtime_error("ERROR: MaxPool2DGradFn: Output gradient spatial dimensions mismatch");
		}

		const size_t outputHeight = gradOutput.Dims()[2];
		const size_t outputWidth = gradOutput.Dims()[3];

		if (gradOutput.Dims()[0] != batchSize) {
			throw std::runtime_error("ERROR: MaxPool2DGradFn: Output gradient batch size mismatch");
		}

		if (gradOutput.Dims()[1] != channels) {
			throw std::runtime_error("ERROR: MaxPool2DGradFn: Output gradient channel size mismatch");
		}

		Memory::ArenaAllocator& allocator = input.GetAllocator();

		TensorCore::Tensor<T> gradInput{ {batchSize, channels, inputHeight, inputWidth}, allocator };
		gradInput.Fill(static_cast<T>(0));

		if (input.IsContiguous() && indices.IsContiguous()) {
			const T* gradOutputData = gradOutput.Data();
			const T* indicesData = indices.Data();
			T* gradInputData = gradInput.Data();

			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t c = 0; c < channels; ++c) {
					for (size_t oh = 0; oh < outputHeight; ++oh) {
						for (size_t ow = 0; ow < outputWidth; ++ow) {
							size_t gradIndex = ((n * channels + c) * outputHeight + oh) * outputWidth + ow;

							T grad = gradOutputData[gradIndex];
							T maxIdx = indicesData[gradIndex];

							if (maxIdx >= 0) {
								gradInputData[static_cast<size_t>(maxIdx)] += grad;
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
							size_t gradIndex = ((n * channels + c) * outputHeight + oh) * outputWidth + ow;

							T grad = gradOutput[gradIndex];
							T maxIdx = indices[gradIndex];

							if (maxIdx >= 0) {
								gradInput[maxIdx] += grad;
							}
						}
					}
				}
			}
		}

		input.Backward(gradInput);
	}

	template <typename T>
	MaxPool3DGradFn<T>::MaxPool3DGradFn(std::shared_ptr<TensorCore::TensorImpl<T>> input, std::shared_ptr<TensorCore::TensorImpl<T>> indices,
										size_t strideD, size_t strideH, size_t strideW,
										size_t paddingD, size_t paddingH, size_t paddingW,
										size_t dilationD, size_t dilationH, size_t dilationW)
		: GradFn<T>({input, indices}), m_StrideD(strideD), m_StrideH(strideH), m_StrideW(strideW), m_PaddingD(paddingD), m_PaddingH(paddingH), m_PaddingW(paddingW),
		  m_DilationD(dilationD), m_DilationH(dilationH), m_DilationW(dilationW)
	{}

	template <typename T>
	void MaxPool3DGradFn<T>::Backward(const TensorCore::Tensor<T>& gradOutput) {
		if (!this->inputs[0]) {
			throw std::runtime_error("ERROR: MaxPool3DGradFn: Null input");
		}

		if (!this->inputs[0]) {
			throw std::runtime_error("ERROR: MaxPool3DGradFn: Null indices");
		}

		TensorCore::Tensor<T> input{ this->inputs[0] };
		TensorCore::Tensor<T> indices{ this->inputs[1] };

		if (!input.RequiresGrad()) {
			return;
		}

		if (gradOutput.Rank() != 5) {
			throw std::runtime_error("ERROR: MaxPool3DGradFn: Output gradient must have 5 dimensions");
		}

		const size_t batchSize = input.Dims()[0];
		const size_t channels = input.Dims()[1];
		const size_t inputDepth = input.Dims()[2];
		const size_t inputHeight = input.Dims()[3];
		const size_t inputWidth = input.Dims()[4];

		if (gradOutput.Dims()[2] != indices.Dims()[2] || gradOutput.Dims()[3] != indices.Dims()[3] || gradOutput.Dims()[4] != indices.Dims()[4]) {
			throw std::runtime_error("ERROR: MaxPool3DGradFn: Output gradient spatial dimensions mismatch");
		}

		const size_t outputDepth = gradOutput.Dims()[2];
		const size_t outputHeight = gradOutput.Dims()[3];
		const size_t outputWidth = gradOutput.Dims()[4];

		if (gradOutput.Dims()[0] != batchSize) {
			throw std::runtime_error("ERROR: MaxPool3DGradFn: Output gradient batch size mismatch");
		}

		if (gradOutput.Dims()[1] != channels) {
			throw std::runtime_error("ERROR: MaxPool3DGradFn: Output gradient channel size mismatch");
		}

		Memory::ArenaAllocator& allocator = input.GetAllocator();

		TensorCore::Tensor<T> gradInput{ {batchSize, channels, inputDepth, inputHeight, inputWidth}, allocator };
		gradInput.Fill(static_cast<T>(0));

		if (input.IsContiguous() && indices.IsContiguous()) {
			const T* gradOutputData = gradOutput.Data();
			const T* indicesData = indices.Data();
			T* gradInputData = gradInput.Data();

			for (size_t n = 0; n < batchSize; ++n) {
				for (size_t c = 0; c < channels; ++c) {
					for (size_t od = 0; od < outputDepth; ++od) {
						for (size_t oh = 0; oh < outputHeight; ++oh) {
							for (size_t ow = 0; ow < outputWidth; ++ow) {
								size_t gradIndex = (((n * channels + c) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;

								T grad = gradOutputData[gradIndex];
								T maxIdx = indicesData[gradIndex];

								if (maxIdx >= 0) {
									gradInputData[static_cast<size_t>(maxIdx)] += grad;
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
								size_t gradIndex = (((n * channels + c) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;

								T grad = gradOutput[gradIndex];
								T maxIdx = indices[gradIndex];

								if (maxIdx >= 0) {
									gradInput[maxIdx] += grad;
								}
							}
						}
					}
				}
			}
		}

		input.Backward(gradInput);
	}
}