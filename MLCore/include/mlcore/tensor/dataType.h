// dataType.h
#pragma once
#include <cstdint>
#include <type_traits>

namespace MLCore::TensorCore {
	enum class DataType : uint32_t {
		Float32,
		Float64,
		Int32,
		Int64,
		Unknown,
	};

	template <typename T>
	struct DataTypeTraits {
		static constexpr DataType value = DataType::Unknown;
	};

	template <>
	struct DataTypeTraits<float> {
		static constexpr DataType value = DataType::Float32;
	};

	template <>
	struct DataTypeTraits<double> {
		static constexpr DataType value = DataType::Float64;
	};

	template <>
	struct DataTypeTraits<int32_t> {
		static constexpr DataType value = DataType::Int32;
	};

	template <>
	struct DataTypeTraits<int64_t> {
		static constexpr DataType value = DataType::Int64;
	};

	template <typename T>
	constexpr DataType ExpectedType() {
		return DataTypeTraits<T>::value;
	}
}