// tensor.inl
#include <stdexcept>

namespace MLCore::TensorCore {
	template <typename T>
	inline Tensor<T>::Tensor(const Utils::Shape& shape, Memory::ArenaAllocator& allocator)
		: m_Shape(shape), m_Allocator(&allocator), m_Storage(Memory::MakeStorage<T>(allocator, shape.NumElements()))
	{}

	template <typename T>
	inline Tensor<T>::Tensor(const Tensor<T>& other)
		: m_Shape(other.GetShape()), m_Allocator(const_cast<Memory::ArenaAllocator*>(&(other.GetAllocator()))), m_Storage(Memory::MakeStorage<T>(*m_Allocator, m_Shape.NumElements())) {
		// This for-loop form of allocation can be optimized with std::memcpy
		for (size_t i = 0; i < NumElements(); ++i) {
			(*this)[i] = other[i];
		}
	}

	template <typename T>
	inline Tensor<T>::Tensor(Tensor&& other) noexcept
		: m_Shape(std::move(other.m_Shape)), m_Allocator(other.m_Allocator), m_Storage(std::move(other.m_Storage)), m_Grad(std::move(other.m_Grad)),
		  m_GradFn(std::move(other.m_GradFn)), m_RequiresGrad(other.m_RequiresGrad)
	{}

	template <typename T>
	inline Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
		if (this != &other) {
			m_Shape = std::move(other.m_Shape);
			m_Allocator = other.m_Allocator;
			m_Storage = std::move(other.m_Storage);
			m_Grad = std::move(other.m_Grad);
			m_GradFn = std::move(other.m_GradFn);
			m_RequiresGrad = other.m_RequiresGrad;
		}

		return *this;
	}

	template <typename T>
	inline Tensor<T>::Tensor(std::initializer_list<size_t> dims, Memory::ArenaAllocator& allocator)
		: m_Shape(dims), m_Allocator(&allocator), m_Storage(Memory::MakeStorage<T>(allocator, m_Shape.NumElements()))
	{}

	template <typename T>
	inline Tensor<T>::Tensor(std::vector<size_t> dims, Memory::ArenaAllocator& allocator)
		: m_Shape(dims), m_Allocator(&allocator), m_Storage(Memory::MakeStorage<T>(allocator, m_Shape.NumElements()))
	{}

	template <typename T>
	inline const Utils::Shape& Tensor<T>::GetShape() const {
		return m_Shape;
	}

	template <typename T>
	inline size_t Tensor<T>::NumElements() const {
		return m_Shape.NumElements();
	}

	template <typename T>
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

	template <typename T>
	Memory::ArenaAllocator& Tensor<T>::GetAllocator() {
		return *m_Allocator;
	}

	template <typename T>
	const Memory::ArenaAllocator& Tensor<T>::GetAllocator() const {
		return *m_Allocator;
	}

	template <typename T>
	inline T* Tensor<T>::begin() {
		return m_Storage.Data();
	}

	template <typename T>
	inline T* Tensor<T>::end() {
		return m_Storage.Data() + NumElements();
	}

	template <typename T>
	inline const T* Tensor<T>::begin() const {
		return m_Storage.Data();
	}

	template <typename T>
	inline const T* Tensor<T>::end() const {
		return m_Storage.Data() + NumElements();
	}

	template <typename T>
	inline T& Tensor<T>::operator[](size_t i) {
		#ifdef ML_CORE_DEBUG
			if (i >= m_Storage.Size()) {
				throw std::out_of_range("ERROR: Tensor linear index out of bounds");
			}
		#endif

		return m_Storage.Data()[i];
	}

	template <typename T>
	inline const T& Tensor<T>::operator[](size_t i) const {
		#ifdef ML_CORE_DEBUG
			if (i >= m_Storage.Size()) {
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
	bool Tensor<T>::RequiresGrad() const {
		return m_RequiresGrad;
	}

	template <typename T>
	bool Tensor<T>::HasGrad() const {
		return m_Grad != nullptr;
	}

	template <typename T>
	void Tensor<T>::ZeroGrad() {
		if (m_Grad) {
			for (size_t i = 0; i < m_Grad->NumElements(); ++i) {
				(*m_Grad)[i] = static_cast<T>(0);
			}
		}
	}

	template <typename T>
	bool Tensor<T>::IsLeaf() const {
		return m_GradFn == nullptr;
	}

	template <typename T>
	void Tensor<T>::SetRequiresGrad(bool require) {
		m_RequiresGrad = require;
	}

	template <typename T>
	Tensor<T>* Tensor<T>::Grad() {
		return m_Grad.get();
	}

	template <typename T>
	const Tensor<T>* Tensor<T>::Grad() const {
		return m_Grad.get();
	}

	template <typename T>
	Tensor<T> Tensor<T>::Detach() {
		Tensor<T> out{ m_Shape, m_Allocator };

		for (size_t i = 0; i < NumElements(); ++i) {
			out[i] = (*this)[i];
		}

		return out;
	}

	template <typename T>
	AutoGrad::GradFn<T>* Tensor<T>::GradFn() {
		return m_GradFn.get();
	}

	template <typename T>
	const AutoGrad::GradFn<T>* Tensor<T>::GradFn() const {
		return m_GradFn.get();
	}

	template <typename T>
	void Tensor<T>::SetGradFn(AutoGrad::GradFn<T>* gradFn) {
		m_GradFn.reset(gradFn);
	}

	template <typename T>
	void Tensor<T>::AccumulateGrad(const Tensor<T>& gradInput) {
		if (!m_RequiresGrad) {
			return;
		}

		if (!m_Grad) {
			auto& allocator = const_cast<Memory::ArenaAllocator&>(gradInput.GetAllocator());
			m_Grad = std::make_unique<Tensor<T>>(gradInput.GetShape(), allocator);
			m_Grad->Fill(static_cast<T>(0));

			for (size_t i = 0; i < gradInput.NumElements(); ++i) {
				(*m_Grad)[i] += gradInput[i];
			}
		}
		else {
			for (size_t i = 0; i < gradInput.NumElements(); ++i) {
				(*m_Grad)[i] += gradInput[i];
			}
		}
	}

	template <typename T>
	void Tensor<T>::Backward(const Tensor<T>& gradOutput) {
		if (!m_RequiresGrad) {
			return;
		}
		
		AccumulateGrad(gradOutput);

		if (m_Visited) {
			return;
		}

		m_Visited = true;

		if (m_GradFn) {
			m_GradFn->Backward(gradOutput);
		}
	}
	
	template <typename T>
	void Tensor<T>::Backward(Memory::ArenaAllocator& allocator) {
		if (!m_RequiresGrad) {
			return;
		}

		Tensor<T> gradOutput{ GetShape(), allocator};
		gradOutput.Fill(static_cast<T>(1));

		if (m_Visited) {
			return;
		}

		m_Visited = true;

		if (m_GradFn) {
			m_GradFn->Backward(gradOutput);
		}
	}
}