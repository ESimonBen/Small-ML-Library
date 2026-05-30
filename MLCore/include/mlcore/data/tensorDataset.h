// tensorDataset.h
#pragma once
#include <mlCore/data/dataset.h>

namespace MLCore::Data {
	template <typename T>
	class TensorDataset : public Dataset<T> {
	public:
		TensorDataset(const TensorCore::Tensor<T>& inputs, const TensorCore::Tensor<T>& targets);

		virtual size_t Size() const override;

		virtual std::pair<TensorCore::Tensor<T>, TensorCore::Tensor<T>> GetItem(size_t index) const override;

	private:
		TensorCore::Tensor<T> m_Inputs;
		TensorCore::Tensor<T> m_Targets;
	};
}

#include "tensorDataset.inl"