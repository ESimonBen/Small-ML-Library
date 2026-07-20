 /// dataType.h
#pragma once
#include <cstdint>

namespace MLCore::TensorCore {
	/// <summary>
	/// Represents a data type identifier with a 32-bit unsigned underlying type.
	/// </summary>
	enum class DataType : uint32_t {
		Float32,
		Float64,
		Int32,
		Int64,
		Unknown,
	};

	/// <summary>
	/// Primary template that associates a C++ type with a DataType enumeration; the default mapping yields DataType::Unknown.
	/// </summary>
	/// <typeparam name="T">The C++ type being described by the trait.</typeparam>
	template <typename T>
	struct DataTypeTraits {
		static constexpr DataType value = DataType::Unknown;
	};

	/// <summary>
	/// Explicit specialization of DataTypeTraits for float that defines the trait value as DataType::Float32.
	/// </summary>
	template <>
	struct DataTypeTraits<float> {
		static constexpr DataType value = DataType::Float32;
	};

	/// <summary>
	/// Specialization of DataTypeTraits for the double type that maps it to DataType::Float64.
	/// </summary>
	template <>
	struct DataTypeTraits<double> {
		static constexpr DataType value = DataType::Float64;
	};

	/// <summary>
	/// Specialization of DataTypeTraits for int32_t that exposes the corresponding DataType enumerator.
	/// </summary>
	template <>
	struct DataTypeTraits<int32_t> {
		static constexpr DataType value = DataType::Int32;
	};

	/// <summary>
	/// Full specialization of DataTypeTraits for int64_t that maps the C++ type int64_t to the DataType::Int64 enumeration value.
	/// </summary>
	template <>
	struct DataTypeTraits<int64_t> {
		static constexpr DataType value = DataType::Int64;
	};

	/// <summary>
	/// Retrieves the DataType associated with the template type T using DataTypeTraits, evaluated at compile time.
	/// </summary>
	/// <typeparam name="T">The type for which the corresponding DataType is obtained.</typeparam>
	/// <returns>The DataType value provided by DataTypeTraits<T>::value (a compile-time constant).</returns>
	template <typename T>
	constexpr DataType ExpectedType() {
		return DataTypeTraits<T>::value;
	}
}