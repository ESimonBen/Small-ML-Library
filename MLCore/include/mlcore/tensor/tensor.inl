// tensor.inl
#include <stdexcept>

namespace MLCore::TensorCore {
	template <typename T>
	inline Tensor<T>::Tensor(const Utils::Shape& shape, Memory::ArenaAllocator& allocator)
		: m_Shape(shape), m_Storage(Memory::MakeStorage<T>(allocator, shape.NumElements())) 
	{}

	template<typename T>
	inline Tensor<T>::Tensor(std::initializer_list<size_t> dims, Memory::ArenaAllocator& allocator)
		: m_Shape(dims), m_Storage(Memory::MakeStorage<T>(allocator, m_Shape.NumElements())) 
	{}

	template<typename T>
	inline Tensor<T>::Tensor(std::vector<size_t> dims, Memory::ArenaAllocator& allocator)
		: m_Shape(dims), m_Storage(Memory::MakeStorage<T>(allocator, m_Shape.NumElements())) 
	{}

	template <typename T>
	inline const Utils::Shape& Tensor<T>::GetShape() const {
		return m_Shape;
	}

	template <typename T>
	inline size_t Tensor<T>::NumElements() const {
		return m_Shape.NumElements();
	}

	template<typename T>
	inline void Tensor<T>::Fill(const T& value) {
		for (size_t i = 0; i < NumElements(); ++i) {
			m_Storage.Data()[i] = value;
		}
	}

	template <typename T>
	inline T* Tensor<T>::Data() {
		return m_Storage.Data();
	}

	template <typename T>
	inline const T* Tensor<T>::Data() const {
		return m_Storage.Data();
	}

	template <typename T>
	inline size_t Tensor<T>::Rank() const {
		return m_Shape.Rank();
	}

	template <typename T>
	inline const std::vector<size_t>& Tensor<T>::Dims() const {
		return m_Shape.Dims();
	}

	template<typename T>
	inline T* Tensor<T>::begin() {
		return m_Storage.Data();
	}

	template<typename T>
	inline T* Tensor<T>::end() {
		return m_Storage.Data() + NumElements();
	}

	template<typename T>
	inline const T* Tensor<T>::begin() const {
		return m_Storage.Data();
	}

	template<typename T>
	inline const T* Tensor<T>::end() const {
		return m_Storage.Data() + NumElements();
	}

	template <typename T>
	inline T& Tensor<T>::operator[](size_t i) {
		#ifdef ML_CORE_DEBUG
			if (i > m_Storage.Size()) {
				throw std::out_of_range("ERROR: Tensor linear index out of bounds");
			}
		#endif

		return m_Storage.Data()[i];
	}

	template <typename T>
	const inline T& Tensor<T>::operator[](size_t i) const {
		#ifdef ML_CORE_DEBUG
			if (i > m_Storage.Size()) {
				throw std::out_of_range("ERROR: Tensor linear index out of bounds");
			}
		#endif

		return m_Storage.Data()[i];
	}


	template <typename T>
	inline T& Tensor<T>::operator()(const std::vector<size_t>& indices) {
		size_t offset = m_Shape.FlattenIndex(indices);

		#ifdef ML_CORE_DEBUG
			if (offset >= m_Storage.Size()) {
				throw std::out_of_range("ERROR: Tensor index out of bounds");
			}
		#endif

		return m_Storage.Data()[offset];
	}

	template <typename T>
	inline const T& Tensor<T>::operator()(const std::vector<size_t>& indices) const {
		size_t offset = m_Shape.FlattenIndex(indices);

		#ifdef ML_CORE_DEBUG
			if (offset >= m_Storage.Size()) {
				throw std::out_of_range("ERROR: Tensor index out of bounds");
			}
		#endif


		return m_Storage.Data()[offset];
	}

	template <typename T>
	template <typename... Indices, typename>
	inline T& Tensor<T>::operator()(Indices... indices) {
		#ifdef ML_CORE_DEBUG
			if (sizeof...(indices) != m_Shape.Rank()) {
				throw std::runtime_error("ERROR: Tensor indexing dimension mismatch");
			}
		#endif

		size_t idx[] = { static_cast<size_t>(indices)... };
		size_t offset = 0;
		const auto& strides = m_Shape.Strides();

		for (size_t i = 0; i < sizeof...(indices); ++i) {
			offset += idx[i] * strides[i];
		}

		#ifdef ML_CORE_DEBUG
			if (offset >= m_Storage.Size()) {
				throw std::out_of_range("ERROR: Tensor index out of bounds");
			}
		#endif


		return m_Storage.Data()[offset];
	}

	template <typename T>
	template <typename... Indices, typename>
	inline const T& Tensor<T>::operator()(Indices... indices) const {
		#ifdef ML_CORE_DEBUG
			if (sizeof...(indices) != m_Shape.Dim()) {
				throw std::runtime_error("ERROR: Tensor indexing dimension mismatch");
			}
		#endif

		size_t idx[] = { static_cast<size_t>(indices)... };
		size_t offset = 0;
		const auto& strides = m_Shape.Strides();

		for (size_t i = 0; i < sizeof...(indices); ++i) {
			offset += idx[i] * strides[i];
		}

		#ifdef ML_CORE_DEBUG
			if (offset >= m_Storage.Size()) {
				throw std::out_of_range("ERROR: Tensor index out of bounds");
			}
		#endif


		return m_Storage.Data()[offset];
	}
}