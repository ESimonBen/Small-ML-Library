// dataLoader.inl
#include <random>
#include <numeric>
#include <algorithm>

namespace MLCore::Data {
	template <typename T>
	DataLoader<T>::DataLoader(const Dataset<T>& dataset, size_t batchSize, bool shuffle)
		: m_Dataset(dataset), m_BatchSize(batchSize), m_Shuffle(shuffle) {
		if (m_BatchSize == 0) {
			throw std::runtime_error("ERROR: Batch size cannot be 0");
		}

		Reset();
	}

	template <typename T>
	void DataLoader<T>::Reset() {
		m_CurrentIndex = 0;

		m_Indices.resize(m_Dataset.Size());

		std::iota(m_Indices.begin(), m_Indices.end(), 0);

		if (m_Shuffle) {
			std::random_device rand;
			std::shuffle(m_Indices.begin(), m_Indices.end(), std::mt19937(rand()));
		}
	}

	template <typename T>
	bool DataLoader<T>::HasNext() const {
		return m_CurrentIndex < m_Indices.size();
	}

	template <typename T>
	std::pair<TensorCore::Tensor<T>, TensorCore::Tensor<T>> DataLoader<T>::Next() {
		if (!HasNext()) {
			throw std::out_of_range("ERROR: No batches remaining");
		}

		size_t end = std::min(m_CurrentIndex + m_BatchSize, m_Indices.size());

		std::vector<TensorCore::Tensor<T>> batchInputs;
		std::vector<TensorCore::Tensor<T>> batchTargets;

		batchInputs.reserve(end - m_CurrentIndex);
		batchTargets.reserve(end - m_CurrentIndex);

		for (size_t i = m_CurrentIndex; i < end; ++i) {
			auto [x, y] = m_Dataset.GetItem(m_Indices[i]);

			batchInputs.push_back(x);
			batchTargets.push_back(y);
		}

		m_CurrentIndex = end;

		return { TensorCore::Tensor<T>::Concat(batchInputs), TensorCore::Tensor<T>::Concat(batchTargets) };
	}
}