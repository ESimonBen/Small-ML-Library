// dataLoader.h
#pragma once
#include <vector>
#include <mlCore/data/dataset.h>

namespace MLCore::Data {
	template <typename T>
	class DataLoader {
	public:
		DataLoader(const Dataset<T>& dataset, size_t batchSize, bool shuffle = true);

		void Reset();

		bool HasNext() const;

		std::pair<TensorCore::Tensor<T>, TensorCore::Tensor<T>> Next();

	private:
		const Dataset<T>& m_Dataset;
		size_t m_BatchSize;
		bool m_Shuffle;
		size_t m_CurrentIndex = 0;
		std::vector<size_t> m_Indices;
	};
}

#include "dataLoader.inl"