// checkpoint.h
#pragma once
#include <fstream>
#include <mlCore/module/module.h>
#include <mlCore/training/trainer.h>
#include <mlCore/optimizers/optimizer.h>
#include <mlCore/schedulers/lrScheduler.h>
#include <mlCore/serialization/binaryArchive.h>

namespace MLCore::Serialization {
	enum class Section : uint8_t {
		Optimizer = 1,
		Scheduler = 2,
		Trainer = 3,
		End = 255
	};

	class Checkpoint {
	public:
		template <typename T>
		static void Save(const NN::Module<T>& model, const std::string& path, const Optimizers::Optimizer<T>* opt = nullptr,
						 const Schedulers::LRScheduler<T>* scheduler = nullptr, const Training::TrainerState<T>* state = nullptr);

		template <typename T>
		static void Load(NN::Module<T>& model, const std::string& path, Optimizers::Optimizer<T>* opt = nullptr,
						 Schedulers::LRScheduler<T>* scheduler = nullptr, Training::TrainerState<T>* state = nullptr);

	private:
		template <typename T>
		static void SaveV1(const NN::Module<T>& model, BinaryWriter& writer);

		template <typename T>
		static void LoadV1(NN::Module<T>& model, BinaryReader& reader);

		template <typename T>
		static void SaveV2(const NN::Module<T>& model, BinaryWriter& writer);

		template <typename T>
		static void LoadV2(NN::Module<T>& model, BinaryReader& reader);

		template <typename T>
		static void SaveV3(const NN::Module<T>& model, BinaryWriter& writer, const Optimizers::Optimizer<T>* opt, const Schedulers::LRScheduler<T>* scheduler, const Training::TrainerState<T>* state);

		template <typename T>
		static void LoadV3(NN::Module<T>& model, BinaryReader& reader, Optimizers::Optimizer<T>* opt, Schedulers::LRScheduler<T>* scheduler, Training::TrainerState<T>* state);

	private:
		static constexpr uint64_t MAGIC_NUMBER = 0x4D4C434F5245ULL; // "MLCORE" in hexadecimal
		static constexpr uint32_t FORMAT_VERSION = 3; // Most recent developed version
	};
}

#include "checkpoint.inl"