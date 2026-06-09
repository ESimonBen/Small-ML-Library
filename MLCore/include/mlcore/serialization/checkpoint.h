// checkpoint.h
#pragma once
#include <fstream>
#include <mlCore/module/module.h>

namespace MLCore::Serialization {
	class Checkpoint {
	public:
		template <typename T>
		static void Save(const NN::Module<T>& model, const std::string& path);

		template <typename T>
		static void Load(NN::Module<T>& model, const std::string& path);

	private:
		template <typename T>
		static void Write(std::ofstream& out, const T& data);

		template <typename T>
		static void Read(std::ifstream& in, T& data);

		template <typename T>
		static void WriteArray(std::ofstream& out, const T* data, size_t count);

		template <typename T>
		static void ReadArray(std::ofstream& in, T* data, size_t count);

		template <typename T>
		static void SaveV1(const NN::Module<T>& model, std::ofstream& out);

		template <typename T>
		static void LoadV1(NN::Module<T>& model, std::ifstream& in);

	private:
		static constexpr uint64_t MAGIC_NUMBER = 0x4D4C434F5245ULL; // "MLCORE" in hexadecimal
		static constexpr uint32_t FORMAT_VERSION = 1;
	};
}

#include "checkpoint.inl"