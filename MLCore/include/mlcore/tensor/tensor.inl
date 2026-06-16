// tensor.inl
#include <stdexcept>

namespace MLCore::TensorCore {
	template <typename T>
	inline Tensor<T>::Tensor(const Utils::Shape& shape, Memory::ArenaAllocator& allocator) {
		auto storage = Memory::MakeStorage<T>(allocator, shape.NumElements());

		m_Impl = std::make_shared<Impl>( shape, std::move(storage), &allocator, 0, false, nullptr, nullptr);
	}

	template <typename T>
	inline Tensor<T>::Tensor(std::initializer_list<size_t> dims, Memory::ArenaAllocator& allocator)
		: Tensor(Utils::Shape{dims}, allocator)
	{}

	template <typename T>
	inline Tensor<T>::Tensor(std::vector<size_t> dims, Memory::ArenaAllocator& allocator)
		: Tensor(Utils::Shape{dims}, allocator)
	{}

	template <typename T>
	inline Tensor<T>::Tensor(std::shared_ptr<Impl> impl)
		: m_Impl(std::move(impl))
	{}

	template <typename T>
	inline Tensor<T> Tensor<T>::Clone() const {
		Tensor<T> out{ m_Impl->shape, (*m_Impl->allocator) };
		size_t size = NumElements();

		for (size_t i = 0; i < size; ++i) {
			out[i] = (*this)[i];
		}

		return out;
	}

	template <typename T>
	inline Tensor<T> Tensor<T>::Detach() const {
		auto newImpl = std::make_shared<Impl>(
			m_Impl->shape,
			m_Impl->storage,              // shared storage (shallow copy OK)
			m_Impl->allocator,
			m_Impl->offset,
			false,                        // requiresGrad = false
			nullptr,
			nullptr
		);

		// This does a std::move of this shared_ptr, which essentially creates a way to view the data without creating it
		return Tensor<T>{newImpl};
	}

	template <typename T>
	inline const Utils::Shape& Tensor<T>::GetShape() const {
		return m_Impl->shape;
	}

	template <typename T>
	inline size_t Tensor<T>::NumElements() const {
		return m_Impl->shape.NumElements();
	}

	template <typename T>
	inline void Tensor<T>::Fill(const T& value) {
		size_t size = NumElements();
		for (size_t i = 0; i < size; ++i) {
			m_Impl->storage.Data()[m_Impl->offset + i] = value;
		}
	}

	template <typename T>
	inline T* Tensor<T>::Data() {
		return m_Impl->storage.Data() + m_Impl->offset;
	}

	template <typename T>
	inline const T* Tensor<T>::Data() const {
		return m_Impl->storage.Data() + m_Impl->offset;
	}

	template <typename T>
	inline size_t Tensor<T>::Rank() const {
		return m_Impl->shape.Rank();
	}

	template <typename T>
	inline const std::vector<size_t>& Tensor<T>::Dims() const {
		return m_Impl->shape.Dims();
	}

	template <typename T>
	Memory::ArenaAllocator& Tensor<T>::GetAllocator() {
		return *(m_Impl->allocator);
	}

	template <typename T>
	Memory::ArenaAllocator& Tensor<T>::GetAllocator() const {
		return *(m_Impl->allocator);
	}

	template <typename T>
	std::shared_ptr<TensorImpl<T>> Tensor<T>::GetImpl() const {
		return m_Impl;
	}

	template <typename T>
	inline T* Tensor<T>::begin() {
		return m_Impl->storage.Data() + m_Impl->offset;
	}

	template <typename T>
	inline T* Tensor<T>::end() {
		return m_Impl->storage.Data() + m_Impl->offset + NumElements();
	}

	template <typename T>
	inline const T* Tensor<T>::begin() const {
		return m_Impl->storage.Data() + m_Impl->offset;
	}

	template <typename T>
	inline const T* Tensor<T>::end() const {
		return m_Impl->storage.Data() + m_Impl->offset + NumElements();
	}

	template <typename T>
	inline T& Tensor<T>::operator[](size_t i) {
		if (i >= NumElements()) {
			throw std::out_of_range("ERROR: Tensor linear index out of bounds");
		}

		return m_Impl->storage.Data()[i + m_Impl->offset];
	}

	template <typename T>
	inline const T& Tensor<T>::operator[](size_t i) const {
		if (i >= NumElements()) {
			throw std::out_of_range("ERROR: Tensor linear index out of bounds");
		}

		return m_Impl->storage.Data()[i + m_Impl->offset];
	}


	template <typename T>
	inline T& Tensor<T>::operator()(const std::vector<size_t>& indices) {
		size_t offset = m_Impl->shape.FlattenIndex(indices);

		if (offset >= NumElements()) {
			throw std::out_of_range("ERROR: Tensor index out of bounds");
		}

		return m_Impl->storage.Data()[m_Impl->offset + offset];
	}

	template <typename T>
	inline const T& Tensor<T>::operator()(const std::vector<size_t>& indices) const {
		size_t offset = m_Impl->shape.FlattenIndex(indices);

		if (offset >= NumElements()) {
			throw std::out_of_range("ERROR: Tensor index out of bounds");
		}

		return m_Impl->storage.Data()[m_Impl->offset + offset];
	}

	template <typename T>
	template <typename... Indices, typename>
	inline T& Tensor<T>::operator()(Indices... indices) {
		if (sizeof...(indices) != m_Impl->shape.Rank()) {
			throw std::runtime_error("ERROR: Tensor indexing dimension mismatch");
		}

		size_t idx[] = { static_cast<size_t>(indices)... };
		size_t offset = 0;
		const auto& strides = m_Impl->shape.Strides();

		for (size_t i = 0; i < sizeof...(indices); ++i) {
			offset += idx[i] * strides[i];
		}

		if (offset >= NumElements()) {
			throw std::out_of_range("ERROR: Tensor index out of bounds");
		}

		return m_Impl->storage.Data()[m_Impl->offset + offset];
	}

	template <typename T>
	template <typename... Indices, typename>
	inline const T& Tensor<T>::operator()(Indices... indices) const {
		if (sizeof...(indices) != m_Impl->shape.Rank()) {
			throw std::runtime_error("ERROR: Tensor indexing dimension mismatch");
		}

		size_t idx[] = { static_cast<size_t>(indices)... };
		size_t offset = 0;
		const auto& strides = m_Impl->shape.Strides();

		for (size_t i = 0; i < sizeof...(indices); ++i) {
			offset += idx[i] * strides[i];
		}

		if (offset >= NumElements()) {
			throw std::out_of_range("ERROR: Tensor index out of bounds");
		}

		return m_Impl->storage.Data()[m_Impl->offset + offset];
	}

	template <typename T>
	bool Tensor<T>::RequiresGrad() const {
		return m_Impl->requiresGrad;
	}

	template <typename T>
	bool Tensor<T>::HasGrad() const {
		return m_Impl->grad != nullptr;
	}

	template <typename T>
	void Tensor<T>::ZeroGrad() {
		if (!m_Impl->grad) {
			return;
		}

		Tensor<T> gradTensor{ m_Impl->grad };

		if (m_Impl->grad) {
			size_t size = gradTensor.NumElements();
			for (size_t i = 0; i < size; ++i) {
				gradTensor[i] = static_cast<T>(0);
			}
		}
	}

	template <typename T>
	void Tensor<T>::SetRequiresGrad(bool require) {
		m_Impl->requiresGrad = require;
	}

	template <typename T>
	Tensor<T> Tensor<T>::Grad() {
		if (!m_Impl->grad) {
			Tensor<T> zero{ m_Impl->shape, *(m_Impl->allocator) };
			zero.Fill(static_cast<T>(0));
			return zero;
		}

		return Tensor<T>{m_Impl->grad};
	}

	template <typename T>
	const Tensor<T> Tensor<T>::Grad() const {
		if (!m_Impl->grad) {
			throw std::runtime_error("ERROR: Gradient doesn't exist");
		}

		return Tensor<T>{m_Impl->grad};
	}

	template <typename T>
	std::shared_ptr<AutoGrad::GradFn<T>> Tensor<T>::GradFn() {
		return m_Impl->gradFn;
	}

	template <typename T>
	const std::shared_ptr<AutoGrad::GradFn<T>> Tensor<T>::GradFn() const {
		return m_Impl->gradFn;
	}

	template <typename T>
	void Tensor<T>::SetGradFn(std::shared_ptr<AutoGrad::GradFn<T>> gradFn) {
		m_Impl->gradFn = std::move(gradFn);
	}

	template <typename T>
	void Tensor<T>::AccumulateGrad(const Tensor<T>& gradInput) {
		if (!m_Impl->requiresGrad) {
			return;
		}

		if (!m_Impl->grad) {
			Tensor<T> grad{ m_Impl->shape, (*m_Impl->allocator) };
			grad.Fill(static_cast<T>(0));

			m_Impl->grad = grad.GetImpl();
		}

		auto gradTensor = Tensor<T>{ m_Impl->grad };
		size_t size = gradInput.NumElements();

		for (size_t i = 0; i < size; ++i) {
			gradTensor[i] += gradInput[i];
		}
	}

	// Should get rid of this
	template<typename T>
	inline void Tensor<T>::Backward() {
		if (!m_Impl->requiresGrad) {
			return;
		}

		if (NumElements() != 1) {
			throw std::runtime_error("ERROR: Backward() without gradOutput only allowed for scalar tensors");
		}

		Tensor<T> gradOutput{ m_Impl->shape, (*m_Impl->allocator) };
		gradOutput.Fill(static_cast<T>(1));

		Backward(gradOutput);
	}

	template <typename T>
	void Tensor<T>::Backward(const Tensor<T>& gradOutput) {
		if (!m_Impl->requiresGrad) {
			return;
		}

		AccumulateGrad(gradOutput);

		if (m_Impl->gradFn) {
			m_Impl->gradFn->Backward(gradOutput, *(m_Impl->allocator));
		}
	}

	template <typename T>
	Tensor<T> Tensor<T>::SliceRows(size_t start, size_t end) const{
		if (Rank() < 1) {
			throw std::runtime_error("Cannot slice scalar tensor");
		}

		size_t rows = Dims()[0];

		if (start >= rows || end > rows || start >= end) {
			throw std::out_of_range("Invalid slice range");
		}

		std::vector<size_t> newDims = Dims();
		newDims[0] = end - start;

		auto newImpl = std::make_shared<Impl>(
			Utils::Shape{newDims},
			m_Impl->storage,
			m_Impl->allocator,
			m_Impl->offset + start * m_Impl->shape.Strides()[0],
			false,
			nullptr,
			nullptr
		);

		return Tensor<T>{newImpl};
	}

	template <typename T>
	Tensor<T> Tensor<T>::Concat(const std::vector<Tensor<T>>& tensors) {
		if (tensors.empty()) {
			throw std::runtime_error("ERROR: Cannot concatenate empty tensors");
		}

		// Reference tensor
		const Tensor<T>& firstTensor = tensors[0];

		const auto& baseDims = firstTensor.Dims();
		size_t rank = firstTensor.Rank();

		if (rank == 0) {
			throw std::runtime_error("ERROR: Cannot concatenate scalar tensors");
		}

		for (size_t i = 1; i < tensors.size(); ++i) {
			// Current Tensor
			const Tensor<T>& currentTensor = tensors[i];

			if (&currentTensor.GetAllocator() != &firstTensor.GetAllocator()) {
				throw std::runtime_error("ERROR: Tensor allocator mismatch in concatenation");
			}

			if (currentTensor.Rank() != rank) {
				throw std::runtime_error("ERROR: Tensor rank mismatch in concatenation");
			}

			const auto& dims = currentTensor.Dims();

			for (size_t d = 1; d < rank; ++d) {
				if (dims[d] != baseDims[d]) {
					throw std::runtime_error("ERROR: Tensor shape mismatch in concatenation");
				}
			}
		}

		// Find output shape
		std::vector<size_t> outDims = baseDims;
		outDims[0] = 0;

		for (const Tensor<T>& tensor : tensors) {
			outDims[0] += tensor.Dims()[0];
		}

		// Copy data into output tensor
		Tensor<T> result{ outDims, firstTensor.GetAllocator() };
		size_t writeOffset = 0;

		for (const Tensor<T>& tensor : tensors) {
			size_t size = tensor.NumElements();

			for (size_t i = 0; i < size; ++i) {
				result[writeOffset + i] = tensor[i];
			}

			writeOffset += size;
		}

		return result;
	}
}