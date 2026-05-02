// parameter.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::NN {
	template <typename T>
	struct Parameter {
		TensorCore::Tensor<T> data;

		Parameter() = default;

		explicit Parameter(const TensorCore::Tensor<T>& tensor)
			: data(tensor)
		{}

		TensorCore::Tensor<T>& Data() {
			return data;
		}

		const TensorCore::Tensor<T>& Data() const {
			return data;
		}

		T* RawData() {
			return data.Data();
		}

		const T* RawData() const {
			return data.Data();
		}
	};
}