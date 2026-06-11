// binaryArchive.h
#pragma once
#include <fstream>

namespace MLCore::Serialization {
	class BinaryWriter {
	public:
		explicit BinaryWriter(std::ofstream& out);

		template <typename T>
		void Write(const T& data);

		template <typename T>
		void WriteArray(const T* data, size_t count);

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

	private:
		std::ifstream& m_In;
	};
}

#include "binaryArchive.inl"