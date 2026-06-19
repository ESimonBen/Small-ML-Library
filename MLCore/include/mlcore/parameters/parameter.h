 /// parameter.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::NN {
	using ParamID = uint64_t;

	/// <summary>
	/// Represents a named tensor parameter with a unique identifier and provides access to its underlying tensor data.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensor (e.g., float, double, int).</typeparam>
	template <typename T>
	struct Parameter {
		TensorCore::Tensor<T> data; /// A Tensor that stores values of type T.
		const ParamID id; /// An ID for the Parameter

		/// <summary>
		/// Parameter's default (parameterless) constructor that initializes the member 'id' by calling NextID().
		/// </summary>
		Parameter()
			: id(NextID())
		{}

		/// <summary>
		/// Explicit constructor that initializes a Parameter from a Tensor, copying the tensor into the data member and assigning a new identifier.
		/// </summary>
		/// <param name="tensor">Reference to a TensorCore::Tensor<T> used to initialize the Parameter's data member.</param>
		explicit Parameter(const TensorCore::Tensor<T>& tensor)
			: data(tensor), id(NextID())
		{}

		Parameter(const Parameter&) = delete;
		Parameter& operator=(const Parameter&) = delete;

		Parameter(Parameter&&) = delete;
		Parameter& operator=(Parameter&&) = delete;

		/// <summary>
		/// Returns a sequential ParamID value, incrementing an internal static counter each call.
		/// </summary>
		/// <returns>A ParamID value drawn from an internal counter, starting at 0 and increasing by 1 on each call. The function returns the current counter value and then increments it. Not safe for concurrent use without external synchronization.</returns>
		static ParamID NextID() {
			static ParamID counter = 0;
			return counter++;
		}

		/// <summary>
		/// Returns a mutable reference to the parameter's underlying data member.
		/// </summary>
		/// <returns>A non-const reference to the internal TensorCore::Tensor<T> data member. Modifying the returned reference will modify the object's internal data.</returns>
		TensorCore::Tensor<T>& Data() {
			return data;
		}

		/// <summary>
		/// Returns a const reference to the stored tensor data.
		/// </summary>
		/// <returns>A const reference to the underlying TensorCore::Tensor<T> held by the object.</returns>
		const TensorCore::Tensor<T>& Data() const {
			return data;
		}

		/// <summary>
		/// Returns a raw pointer to the underlying data buffer.
		/// </summary>
		/// <returns>A pointer to the first element of the underlying data (T*). May be nullptr if the buffer is empty.</returns>
		T* RawData() {
			return data.Data();
		}

		/// <summary>
		/// Returns a pointer to the object's internal data buffer.
		/// </summary>
		/// <returns>A read-only pointer (const T*) to the underlying array of elements. The pointer refers to the internal storage and remains valid until the object is modified or destroyed; it may be null if the container is empty.</returns>
		const T* RawData() const {
			return data.Data();
		}
	};
}