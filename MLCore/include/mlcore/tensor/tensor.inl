// tensor.inl
#include <stdexcept>
#include "tensor.h"

namespace MLCore::TensorCore {
	template <typename T>
	inline Tensor<T>::Tensor(const Utils::Shape& shape, Memory::ArenaAllocator& allocator) {
		auto storage = Memory::MakeStorage<T>(allocator, shape.NumElements());

		m_Impl = std::make_shared<Impl>(Impl{ shape, std::move(storage), &allocator, false, nullptr, nullptr });
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
		for (size_t i = 0; i < NumElements(); ++i) {
			m_Impl->storage.Data()[i] = value;
		}
	}

	template <typename T>
	inline T* Tensor<T>::Data() {
		return m_Impl->storage.Data();
	}

	template <typename T>
	inline const T* Tensor<T>::Data() const {
		return m_Impl->storage.Data();
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
	std::shared_ptr<TensorImpl<T>> Tensor<T>::GetImpl() const {
		return m_Impl;
	}

	template <typename T>
	inline T* Tensor<T>::begin() {
		return m_Impl->storage.Data();
	}

	template <typename T>
	inline T* Tensor<T>::end() {
		return m_Impl->storage.Data() + NumElements();
	}

	template <typename T>
	inline const T* Tensor<T>::begin() const {
		return m_Impl->storage.Data();
	}

	template <typename T>
	inline const T* Tensor<T>::end() const {
		return m_Impl->storage.Data() + NumElements();
	}

	template <typename T>
	inline T& Tensor<T>::operator[](size_t i) {
		#ifdef ML_CORE_DEBUG
			if (i >= m_Impl->storage.Size()) {
				throw std::out_of_range("ERROR: Tensor linear index out of bounds");
			}
		#endif

		return m_Impl->storage.Data()[i];
	}

	template <typename T>
	inline const T& Tensor<T>::operator[](size_t i) const {
		#ifdef ML_CORE_DEBUG
			if (i >= m_Impl->storage.Size()) {
				throw std::out_of_range("ERROR: Tensor linear index out of bounds");
			}
		#endif

		return m_Impl->storage.Data()[i];
	}


	template <typename T>
	inline T& Tensor<T>::operator()(const std::vector<size_t>& indices) {
		size_t offset = m_Impl->shape.FlattenIndex(indices);

		#ifdef ML_CORE_DEBUG
			if (offset >= m_Impl->storage.Size()) {
				throw std::out_of_range("ERROR: Tensor index out of bounds");
			}
		#endif

		return m_Impl->storage.Data()[offset];
	}

	template <typename T>
	inline const T& Tensor<T>::operator()(const std::vector<size_t>& indices) const {
		size_t offset = m_Impl->shape.FlattenIndex(indices);

		#ifdef ML_CORE_DEBUG
			if (offset >= m_Impl->storage.Size()) {
				throw std::out_of_range("ERROR: Tensor index out of bounds");
			}
		#endif

		return m_Impl->storage.Data()[offset];
	}

	template <typename T>
	template <typename... Indices, typename>
	inline T& Tensor<T>::operator()(Indices... indices) {
		#ifdef ML_CORE_DEBUG
			if (sizeof...(indices) != m_Impl->shape.Rank()) {
				throw std::runtime_error("ERROR: Tensor indexing dimension mismatch");
			}
		#endif

		size_t idx[] = { static_cast<size_t>(indices)... };
		size_t offset = 0;
		const auto& strides = m_Impl->shape.Strides();

		for (size_t i = 0; i < sizeof...(indices); ++i) {
			offset += idx[i] * strides[i];
		}

		#ifdef ML_CORE_DEBUG
			if (offset >= m_Impl->storage.Size()) {
				throw std::out_of_range("ERROR: Tensor index out of bounds");
			}
		#endif


		return m_Impl->storage.Data()[offset];
	}

	template <typename T>
	template <typename... Indices, typename>
	inline const T& Tensor<T>::operator()(Indices... indices) const {
		#ifdef ML_CORE_DEBUG
			if (sizeof...(indices) != m_Impl->shape.Rank()) {
				throw std::runtime_error("ERROR: Tensor indexing dimension mismatch");
			}
		#endif

		size_t idx[] = { static_cast<size_t>(indices)... };
		size_t offset = 0;
		const auto& strides = m_Impl->shape.Strides();

		for (size_t i = 0; i < sizeof...(indices); ++i) {
			offset += idx[i] * strides[i];
		}

		#ifdef ML_CORE_DEBUG
			if (offset >= m_Impl->storage.Size()) {
				throw std::out_of_range("ERROR: Tensor index out of bounds");
			}
		#endif


		return m_Impl->storage.Data()[offset];
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
			/*throw std::runtime_error("ERROR: Gradient doesn't exist");*/
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

		// May add this back in TensorImpl
		/*m_Visited = true;*/

		if (m_Impl->gradFn) {
			m_Impl->gradFn->Backward(gradOutput, *(m_Impl->allocator));
		}
	}
}