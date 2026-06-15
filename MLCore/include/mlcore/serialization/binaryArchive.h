// binaryArchive.h
#pragma once
#include <fstream>
#include <mlCore/tensor/tensor.h>

namespace MLCore::Serialization {
	class BinaryWriter {
	public:
		explicit BinaryWriter(std::ofstream& out);

		template <typename T>
		void Write(const T& data);

		template <typename T>
		void WriteArray(const T* data, size_t count);

		template <typename T>
		void WriteTensor(const TensorCore::Tensor<T>& tensor);

	private:
		std::ofstream& m_Out;
	};

	class BinaryReader {
	public:
		explicit BinaryReader(std::ifstream& in);

		template <typename T>
		void Read(T& data);

		template <typename T>
		void ReadArray(T* data, size_t count);

		template <typename T>
		void ReadTensor(TensorCore::Tensor<T>& tensor);

	private:
		std::ifstream& m_In;
	};
}

#include "binaryArchive.inl"