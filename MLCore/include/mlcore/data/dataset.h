// dataset.h
#pragma once
#include <utility>
#include <cstddef>
#include <mlCore/tensor/tensor.h>

namespace MLCore::Data {
	template <typename T>
	class Dataset {
	public:
		virtual ~Dataset() = default;

		virtual size_t Size() const = 0;

		virtual std::pair<TensorCore::Tensor<T>, TensorCore::Tensor<T>> GetItem(size_t index) const = 0;
	};
}