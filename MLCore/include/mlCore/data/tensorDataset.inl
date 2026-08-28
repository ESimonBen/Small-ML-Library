 /// tensorDataset.inl

namespace MLCore::Data {
	template <typename T>
	inline TensorDataset<T>::TensorDataset(const TensorCore::Tensor<T>& inputs, const TensorCore::Tensor<T>& targets)
		: m_Inputs(inputs), m_Targets(targets) {
		if (m_Inputs.IsEmpty() || m_Targets.IsEmpty()) {
			throw std::runtime_error("ERROR: Empty datsets");
		}

		if (m_Inputs.Dims()[0] != m_Targets.Dims()[0]) {
			throw std::runtime_error("ERROR: Dataset sample mismatch");
		}

		#ifdef MLCORE_DEBUG
			m_Inputs.GetAllocator().RegisterPersistent(m_Inputs.Data(), m_Inputs.NumElements() * sizeof(T));
			m_Targets.GetAllocator().RegisterPersistent(m_Targets.Data(), m_Targets.NumElements() * sizeof(T));
		#endif
	}
	
	template <typename T>
	inline size_t TensorDataset<T>::Size() const {
		return m_Inputs.Dims()[0];
	}

	template <typename T>
	inline std::pair<TensorCore::Tensor<T>, TensorCore::Tensor<T>> TensorDataset<T>::GetItem(size_t index) const {
		return { m_Inputs.SliceRows(index, index + 1), m_Targets.SliceRows(index, index + 1) };
	}
}