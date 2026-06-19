 /// binaryArchive.h
#pragma once
#include <fstream>
#include <mlCore/tensor/tensor.h>

namespace MLCore::Serialization {
	/// <summary>
	/// BinaryWriter writes values, arrays, and tensors in binary form to a provided output file stream.
	/// </summary>
	class BinaryWriter {
	public:
		/// <summary>
		/// Constructs a BinaryWriter that uses the provided output file stream.
		/// </summary>
		/// <param name="out">Reference to an std::ofstream that the BinaryWriter will use for output. The stream must remain valid for the lifetime of the BinaryWriter.</param>
		explicit BinaryWriter(std::ofstream& out);

		/// <summary>
		/// Writes an object to the underlying output stream as raw bytes. The function writes sizeof(T) bytes from the object's memory representation and throws std::runtime_error if the write operation fails.
		/// </summary>
		/// <typeparam name="T">Type of the object to write. Should be trivially copyable (or otherwise have a well-defined memory representation) for safe binary writing.</typeparam>
		/// <param name="data">Const reference to the object to write. Its raw memory representation (sizeof(T) bytes) is written to the stream.</param>
		template <typename T>
		void Write(const T& data);

		/// <summary>
		/// Writes an array of elements to the underlying output stream as raw bytes. Throws std::runtime_error if the write fails.
		/// </summary>
		/// <typeparam name="T">The element type. Elements are written using their raw memory representation (sizeof(T) * count bytes).</typeparam>
		/// <param name="data">Pointer to the first element of the array to write. Must point to at least 'count' elements; behavior is undefined if null when count > 0.</param>
		/// <param name="count">Number of elements of type T to write.</param>
		template <typename T>
		void WriteArray(const T* data, size_t count);

		/// <summary>
		/// Writes a tensor's metadata (number of elements, rank, dimensions) followed by its element data to the BinaryWriter's output stream.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor.</typeparam>
		/// <param name="tensor">The tensor to serialize; its element count, rank, dimension array, and data array are written.</param>
		template <typename T>
		void WriteTensor(const TensorCore::Tensor<T>& tensor);

	private:
		std::ofstream& m_Out; /// Reference to an output file stream used for writing.
	};

	/// <summary>
	/// BinaryReader provides utilities to read binary data from a given input file stream into values, arrays, or tensor objects.
	/// </summary>
	class BinaryReader {
	public:
		/// <summary>
		/// Initializes a BinaryReader to use the provided input file stream.
		/// </summary>
		/// <param name="in">A reference to an std::ifstream that the BinaryReader will use for reading binary data.</param>
		explicit BinaryReader(std::ifstream& in);

		/// <summary>
		/// Reads sizeof(T) raw bytes from the BinaryReader's internal input stream into the provided object. Throws std::runtime_error if the read fails. T should be a trivially copyable type.
		/// </summary>
		/// <typeparam name="T">The type of the object to read. Must be trivially copyable because the function performs a raw byte read into the object.</typeparam>
		/// <param name="data">Reference to the object that will be overwritten with the bytes read from the stream. The function reads exactly sizeof(T) bytes into this object.</param>
		template <typename T>
		void Read(T& data);

		/// <summary>
		/// Reads 'count' elements of type T from the binary input stream into the provided buffer.
		/// </summary>
		/// <typeparam name="T">Element type to read. Should be trivially copyable because the function reads raw bytes into memory.</typeparam>
		/// <param name="data">Pointer to a buffer with space for count elements of type T. Must be valid and non-null.</param>
		/// <param name="count">Number of elements to read.</param>
		template <typename T>
		void ReadArray(T* data, size_t count);

		/// <summary>
		/// Reads tensor metadata (element count, rank, and dimensions) and element data from the binary stream into the provided tensor, validating that the stored size, rank, and shape match the tensor.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensor (for example, float or double).</typeparam>
		/// <param name="tensor">Reference to the tensor to populate. The method reads the number of elements, rank, dimensions, and the element data into this tensor. If the stored size, rank, or shape do not match the tensor, a std::runtime_error is thrown.</param>
		template <typename T>
		void ReadTensor(TensorCore::Tensor<T>& tensor);

	private:
		std::ifstream& m_In; /// Reference to an input file stream used for reading.
	};
}

#include "binaryArchive.inl"