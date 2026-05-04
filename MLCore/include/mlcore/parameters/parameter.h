// parameter.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::NN {
	using ParamID = uint64_t;

	template <typename T>
	struct Parameter {
		TensorCore::Tensor<T> data;
		const ParamID id;

		Parameter()
			: id(NextID())
		{}

		explicit Parameter(const TensorCore::Tensor<T>& tensor)
			: data(tensor), id(NextID())
		{}

		Parameter(const Parameter&) = delete;
		Parameter& operator=(const Parameter&) = delete;

		Parameter(Parameter&&) = delete;
		Parameter& operator=(Parameter&&) = delete;

		static ParamID NextID() {
			static ParamID counter = 0;
			return counter++;
		}

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

		//Parameter Clone() const {
		//	Parameter copy = *this;
		//	copy.id = NextID();
		//	return copy;
		//}

		//Parameter CloneFrom(const Parameter& other) {
		//	Parameter clone; // New ID
		//	clone.data = other.data;
		//	return clone;
		//}
	};
}