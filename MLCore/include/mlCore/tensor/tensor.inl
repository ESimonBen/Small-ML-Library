 /// tensor.inl
#include <stdexcept>
#include <mlCore/runtime/context.h>

namespace MLCore::TensorCore {
	template <typename T>
	inline Tensor<T>::Tensor(const Utils::Shape& shape, Memory::ArenaAllocator& allocator) {
		auto storage = Memory::MakeStorage<T>(allocator, shape.NumElements());

		m_Impl = std::make_shared<Impl>( shape, Utils::ComputeContiguousStrides(shape), std::move(storage), &allocator);
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
	inline Tensor<T>::Tensor(const Utils::Shape& shape) {
		Memory::ArenaAllocator& allocator = Runtime::MLContext::GetContext().GetAllocator();
		auto storage = Memory::MakeStorage<T>(shape.NumElements());

		m_Impl = std::make_shared<Impl>(shape, Utils::ComputeContiguousStrides(shape), std::move(storage), &allocator);
	}

	template <typename T>
	inline Tensor<T>::Tensor(std::initializer_list<size_t> dims)
		: Tensor(Utils::Shape{dims})
	{}
	
	template <typename T>
	inline Tensor<T>::Tensor(std::vector<size_t> dims)
		: Tensor(Utils::Shape{dims})
	{}

	template <typename T>
	inline Tensor<T>::Tensor(std::shared_ptr<Impl> impl)
		: m_Impl(std::move(impl)) {
		if (!m_Impl) {
			throw std::invalid_argument("ERROR: Tensor implementation cannot be null");
		}
	}
	
	template <typename T>
	inline Tensor<T> Tensor<T>::Zeros(const Utils::Shape& shape) {
		Tensor<T> result{ shape };
		result.Fill(static_cast<T>(0));
		return result;
	}
	
	template <typename T>
	inline Tensor<T> Tensor<T>::Zeros(std::initializer_list<size_t> dims) {
		return Tensor<T>::Zeros(Utils::Shape{dims});
	}

	template <typename T>
	inline Tensor<T> Tensor<T>::Ones(const Utils::Shape& shape) {
		Tensor<T> result{ shape };
		result.Fill(static_cast<T>(1));
		return result;
	}

	template <typename T>
	inline Tensor<T> Tensor<T>::Ones(std::initializer_list<size_t> dims) {
		return Tensor<T>::Ones(Utils::Shape{ dims });
	}

	template <typename T>
	inline Tensor<T> Tensor<T>::Custom(const Utils::Shape& shape, const T& value) {
		Tensor<T> result{ shape };
		result.Fill(value);
		return result;
	}
	
	template <typename T>
	inline Tensor<T> Tensor<T>::Custom(std::initializer_list<size_t> dims, const T& value) {
		return Tensor<T>::Custom(Utils::Shape{ dims }, value);
	}

	template <typename T>
	inline Tensor<T> Tensor<T>::Clone() const {
		Tensor<T> out{ GetShape(), GetAllocator() };
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
			m_Impl->strides,
			m_Impl->storage,
			m_Impl->allocator,
			m_Impl->offset,
			false,
			nullptr,
			nullptr
		);

		/// This does a std::move of this shared_ptr, which essentially creates a way to view the data without copying the data
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
	
	template<typename T>
	inline bool Tensor<T>::IsEmpty() const {
		return NumElements() == 0;
	}
	
	template <typename T>
	inline void Tensor<T>::Fill(const T& value) {
		if (IsEmpty()) {
			throw std::runtime_error("ERROR: Cannot fill empty tensor with a value");
		}

		if (IsContiguous()) {
			std::fill(Data(), Data() + NumElements(), value);
			return;
		}

		size_t size = NumElements();

		for (size_t i = 0; i < size; ++i) {
			(*this)[i] = value;
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
	inline const std::vector<size_t>& Tensor<T>::Strides() const {
		return m_Impl->strides;
	}
	
	template <typename T>
	inline Memory::ArenaAllocator& Tensor<T>::GetAllocator() {
		return *(m_Impl->allocator);
	}
	
	template <typename T>
	inline Memory::ArenaAllocator& Tensor<T>::GetAllocator() const {
		return *(m_Impl->allocator);
	}
	
	template <typename T>
	inline std::shared_ptr<TensorImpl<T>> Tensor<T>::GetImpl() const {
		return m_Impl;
	}
	
	template <typename T>
	inline T* Tensor<T>::begin() {
		if (!IsContiguous()) {
			throw std::runtime_error("ERROR: begin: Tensor must be contiguous before using standard iterators");
		}

		return Data();
	}
	
	template <typename T>
	inline T* Tensor<T>::end() {
		if (!IsContiguous()) {
			throw std::runtime_error("ERROR: end: Tensor must be contiguous before using standard iterators");
		}

		return Data() + NumElements();
	}
	
	template <typename T>
	inline const T* Tensor<T>::begin() const {
		if (!IsContiguous()) {
			throw std::runtime_error("ERROR: begin: Tensor must be contiguous before using standard iterators");
		}

		return Data();
	}
	
	template <typename T>
	inline const T* Tensor<T>::end() const {
		if (!IsContiguous()) {
			throw std::runtime_error("ERROR: end: Tensor must be contiguous before using standard iterators");
		}

		return Data() + NumElements();
	}
	
	template <typename T>
	inline T& Tensor<T>::operator[](size_t i) {
		if (i >= NumElements()) {
			throw std::out_of_range("ERROR: Tensor linear index out of bounds");
		}

		if (IsContiguous()) {
			return Data()[i];
		}

		auto indices = UnflattenIndex(i);

		return (*this)(indices);
	}
	
	template <typename T>
	inline const T& Tensor<T>::operator[](size_t i) const {
		if (i >= NumElements()) {
			throw std::out_of_range("ERROR: Tensor linear index out of bounds");
		}

		if (IsContiguous()) {
			return Data()[i];
		}

		auto indices = UnflattenIndex(i);

		return (*this)(indices);
	}
	
	template <typename T>
	inline T& Tensor<T>::operator()(const std::vector<size_t>& indices) {
		size_t offset = ComputeOffset(indices);
		return m_Impl->storage.Data()[offset];
	}
	
	template <typename T>
	inline const T& Tensor<T>::operator()(const std::vector<size_t>& indices) const {
		size_t offset = ComputeOffset(indices);
		return m_Impl->storage.Data()[offset];
	}

	template <typename T>
	template <typename... Indices, typename>
	inline T& Tensor<T>::operator()(Indices... indices) {
		if (sizeof...(indices) != Rank()) {
			throw std::runtime_error("ERROR: Tensor indexing dimension mismatch");
		}

		std::vector<size_t> idx{ static_cast<size_t>(indices)... };

		return (*this)(idx);
	}

	template <typename T>
	template <typename... Indices, typename>
	inline const T& Tensor<T>::operator()(Indices... indices) const {
		if (sizeof...(indices) != Rank()) {
			throw std::runtime_error("ERROR: Tensor indexing dimension mismatch");
		}

		std::vector<size_t> idx{ static_cast<size_t>(indices)... };

		return (*this)(idx);
	}

	template <typename T>
	inline T& Tensor<T>::AtOffset(size_t physicalOffset) {
		return Data()[physicalOffset];
	}
	
	template <typename T>
	inline const T& Tensor<T>::AtOffset(size_t physicalOffset) const {
		return Data()[physicalOffset];
	}

	template <typename T>
	inline bool Tensor<T>::RequiresGrad() const {
		return m_Impl->requiresGrad;
	}
	
	template <typename T>
	inline bool Tensor<T>::HasGrad() const {
		return m_Impl->grad != nullptr;
	}
	
	template <typename T>
	inline void Tensor<T>::ZeroGrad() {
		if (m_Impl->grad) {
			TensorCore::Tensor<T> gradient{ m_Impl->grad };
			gradient.Fill(static_cast<T>(0));
		}
	}
	
	template <typename T>
	inline void Tensor<T>::SetRequiresGrad(bool require) {
		m_Impl->requiresGrad = require;
	}
	
	template <typename T>
	inline Tensor<T> Tensor<T>::Grad() {
		if (!m_Impl->grad) {
			throw std::runtime_error("ERROR: Gradient doesn't exist");
		}

		return Tensor<T>{m_Impl->grad};
	}
	
	template <typename T>
	inline const Tensor<T> Tensor<T>::Grad() const {
		if (!m_Impl->grad) {
			throw std::runtime_error("ERROR: Gradient doesn't exist");
		}

		return Tensor<T>{m_Impl->grad};
	}
	
	template <typename T>
	inline std::shared_ptr<AutoGrad::GradFn<T>> Tensor<T>::GradFn() {
		return m_Impl->gradFn;
	}
	
	template <typename T>
	inline const std::shared_ptr<AutoGrad::GradFn<T>> Tensor<T>::GradFn() const {
		return m_Impl->gradFn;
	}
	
	template <typename T>
	inline void Tensor<T>::SetGradFn(std::shared_ptr<AutoGrad::GradFn<T>> gradFn) {
		m_Impl->gradFn = std::move(gradFn);
	}
	
	template <typename T>
	inline void Tensor<T>::InitializeGrad() {
		if (!m_Impl->grad) {
			Tensor<T> grad{ m_Impl->shape, *(m_Impl->allocator) };
			grad.Fill(static_cast<T>(0));
			m_Impl->grad = grad.GetImpl();
		}
	}

	template <typename T>
	inline void Tensor<T>::AccumulateGrad(const Tensor<T>& gradInput) {
		if (!m_Impl->requiresGrad) {
			return;
		}

		if (gradInput.GetShape() != GetShape()) {
			throw std::runtime_error("ERROR: AccumulateGrad: Gradient shape mismatch");
		}

		InitializeGrad();

		auto gradTensor = Tensor<T>{ m_Impl->grad };
		size_t size = gradInput.NumElements();

		for (size_t i = 0; i < size; ++i) {
			gradTensor[i] += gradInput[i];
		}
	}
	
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
	inline void Tensor<T>::Backward(const Tensor<T>& gradOutput) {
		if (!m_Impl->requiresGrad) {
			return;
		}

		AccumulateGrad(gradOutput);

		if (m_Impl->gradFn) {
			m_Impl->gradFn->Backward(gradOutput);
		}
	}
	
	template <typename T>
	inline Tensor<T> Tensor<T>::SliceRows(size_t start, size_t end) const {
		if (Rank() < 1) {
			throw std::runtime_error("Cannot slice scalar tensor");
		}

		if (start >= Dims()[0] || end > Dims()[0] || start >= end) {
			throw std::out_of_range("Invalid slice range");
		}

		std::vector<size_t> newDims = Dims();
		newDims[0] = end - start;

		auto newImpl = std::make_shared<Impl>(
			Utils::Shape{newDims},
			m_Impl->strides,
			m_Impl->storage,
			m_Impl->allocator,
			m_Impl->offset + start * m_Impl->strides[0],
			false,
			nullptr,
			nullptr
		);

		return Tensor<T>{newImpl};
	}
	
	template <typename T>
	inline Tensor<T> Tensor<T>::Concat(const std::vector<Tensor<T>>& tensors) {
		if (tensors.empty()) {
			throw std::runtime_error("ERROR: Cannot concatenate empty tensors");
		}

		/// Reference tensor
		const Tensor<T>& firstTensor = tensors[0];

		const auto& baseDims = firstTensor.Dims();
		size_t rank = firstTensor.Rank();

		if (rank == 0) {
			throw std::runtime_error("ERROR: Cannot concatenate scalar tensors");
		}

		for (size_t i = 1; i < tensors.size(); ++i) {
			/// Current Tensor
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

		/// Find output shape
		std::vector<size_t> outDims = baseDims;
		outDims[0] = 0;

		for (const Tensor<T>& tensor : tensors) {
			outDims[0] += tensor.Dims()[0];
		}

		/// Copy data into output tensor
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
	
	template <typename T>
	inline bool Tensor<T>::IsContiguous() const {
		const std::vector<size_t>& dims = m_Impl->shape.Dims();
		const std::vector<size_t>& strides = m_Impl->strides;

		if (dims.empty() || dims.size() == 1) {
			return true;
		}

		size_t dimsSize = dims.size();
		size_t expectedStride = 1;

		for (size_t i = dimsSize; i-- > 0;) {
			if (dims[i] == 0) {
				return true;
			}

			if (strides[i] != expectedStride) {
				return false;
			}

			expectedStride *= dims[i];
		}

		return true;
	}
	
	template <typename T>
	inline size_t Tensor<T>::ComputeOffset(const std::vector<size_t>& indices) const {
		if (indices.size() != Rank()) {
			throw std::runtime_error("ERROR: Tensor index dimension mismatch");
		}

		const std::vector<size_t>& dims = Dims();
		size_t offset = m_Impl->offset;
		size_t rank = Rank();

		for (size_t i = 0; i < rank; ++i) {
			if (indices[i] >= dims[i]) {
				throw std::out_of_range("ERROR: Tensor index out of bounds");
			}

			offset += indices[i] * m_Impl->strides[i];
		}

		return offset;
	}
	
	template <typename T>
	inline std::vector<size_t> Tensor<T>::UnflattenIndex(size_t index) const {
		if (index >= NumElements()) {
			throw std::out_of_range("ERROR: Tensor linear index out of bounds");
		}

		size_t rank = Rank();
		std::vector<size_t> indices(rank);
		const std::vector<size_t>& dims = Dims();

		for (size_t i = rank; i-- > 0;) {
			indices[i] = index % dims[i];
			index /= dims[i];
		}

		return indices;
	}
}