// checkpoint.h
#pragma once
#include <mlCore/module/module.h>

namespace MLCore::Serialization {
	class Checkpoint {
	public:
		template <typename T>
		static void Save(const NN::Module<T>& model, const std::string& path);

		template <typename T>
		static void Load(NN::Module<T>& model, const std::string& path);

	private:
		static constexpr uint32_t MAGIC_NUMBER = 0x4D4C434F5245; // "MLCORE"
		static constexpr uint32_t FORMAT_VERSION = 1;
	};
}

#include "checkpoint.inl"