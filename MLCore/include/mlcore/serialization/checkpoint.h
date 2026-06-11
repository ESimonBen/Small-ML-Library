// checkpoint.h
#pragma once
#include <fstream>
#include <mlCore/module/module.h>
#include <mlCore/serialization/binaryArchive.h>

namespace MLCore::Serialization {
	class Checkpoint {
	public:
		template <typename T>
		static void Save(const NN::Module<T>& model, const std::string& path);

		template <typename T>
		static void Load(NN::Module<T>& model, const std::string& path);

	private:
		template <typename T>
		static void SaveV1(const NN::Module<T>& model, BinaryWriter& writer);

		template <typename T>
		static void LoadV1(NN::Module<T>& model, BinaryReader& reader);

		template <typename T>
		static void SaveV2(const NN::Module<T>& model, BinaryWriter& writer);

		template <typename T>
		static void LoadV2(NN::Module<T>& model, BinaryReader& reader);

	private:
		static constexpr uint64_t MAGIC_NUMBER = 0x4D4C434F5245ULL; // "MLCORE" in hexadecimal
		static constexpr uint32_t FORMAT_VERSION = 2; // Most recent developed version
	};
}

#include "checkpoint.inl"