 /// checkpoint.h
#pragma once
#include <fstream>
#include <mlCore/module/module.h>
#include <mlCore/training/trainer.h>
#include <mlCore/optimizers/optimizer.h>
#include <mlCore/schedulers/lrScheduler.h>
#include <mlCore/serialization/binaryArchive.h>

namespace MLCore::Serialization {
	/// <summary>
	/// Scoped enumeration defining named sections used by the checkpoint system.
	/// </summary>
	enum class Section : uint8_t {
		Optimizer = 1,
		Scheduler = 2,
		Trainer = 3,
		End = 255
	};

	/// <summary>
	/// Utility class for saving and loading neural network checkpoints (model weights and optional optimizer, scheduler, and trainer state) to and from disk using a versioned binary format.
	/// </summary>
	class Checkpoint {
	public:
		/// <summary>
		/// Saves a binary checkpoint for the given model to the specified file path. The function creates any missing parent directories, writes a file header (magic number, format version, and data type), and then delegates to a version-specific save routine. Throws on file open failure or unsupported format version.
		/// </summary>
		/// <typeparam name="T">Element/data type used by the model, optimizer, scheduler, and trainer state (e.g., float, double).</typeparam>
		/// <param name="model">The model whose parameters and metadata will be serialized into the checkpoint.</param>
		/// <param name="path">Filesystem path where the checkpoint file will be written. Parent directories are created if missing.</param>
		/// <param name="opt">Pointer to an optimizer whose state should be saved. May be null if no optimizer state is to be written or if the format version does not include it.</param>
		/// <param name="scheduler">Pointer to a learning-rate scheduler whose state should be saved. May be null if not applicable.</param>
		/// <param name="state">Pointer to trainer state (e.g., epoch, iteration counters) to include in the checkpoint. May be null if not provided.</param>
		template <typename T>
		static void Save(const NN::Module<T>& model, const std::string& path, const Optimizers::Optimizer<T>* opt = nullptr,
						 const Schedulers::LRScheduler<T>* scheduler = nullptr, const Training::TrainerState<T>* state = nullptr);

		/// <summary>
		/// Loads a checkpoint file into the provided model, validating the file header and dispatching to the version-specific loader. Validates magic number and data type and optionally loads optimizer, scheduler, and trainer state for newer checkpoint versions.
		/// </summary>
		/// <typeparam name="T">The data type of the model parameters and checkpoint tensors. The checkpoint's recorded data type must match TensorCore::ExpectedType<T>().</typeparam>
		/// <param name="model">Reference to the NN::Module<T> to populate with weights and state from the checkpoint.</param>
		/// <param name="path">Filesystem path to the checkpoint file to read (opened in binary mode).</param>
		/// <param name="opt">Pointer to an Optimizers::Optimizer<T>; used when loading optimizer state (e.g., for checkpoint version 3). May be nullptr if optimizer state is not present or not needed.</param>
		/// <param name="scheduler">Pointer to a Schedulers::LRScheduler<T>; used when loading learning-rate scheduler state (e.g., for checkpoint version 3). May be nullptr if not needed.</param>
		/// <param name="state">Pointer to a Training::TrainerState<T>; used when loading trainer state (e.g., for checkpoint version 3). May be nullptr if not needed.</param>
		template <typename T>
		static void Load(NN::Module<T>& model, const std::string& path, Optimizers::Optimizer<T>* opt = nullptr,
						 Schedulers::LRScheduler<T>* scheduler = nullptr, Training::TrainerState<T>* state = nullptr);

	private:
		/// <summary>
		/// Serializes the model's parameters into a version 1 binary checkpoint format.
		/// </summary>
		/// <typeparam name="T">Element type of the model's tensors (e.g., float, double).</typeparam>
		/// <param name="model">The model whose parameters will be saved. The function obtains the module's parameter list and serializes each parameter's tensor (size, rank, shape, and data).</param>
		/// <param name="writer">BinaryWriter used to write binary data. The function writes the number of parameters followed by each parameter's tensor via writer.WriteTensor.</param>
		template <typename T>
		static void SaveV1(const NN::Module<T>& model, BinaryWriter& writer);

		/// <summary>
		/// Loads model parameters from a version 1 checkpoint stream into the provided module. It reads the parameter count, verifies it matches the module, and then reads each parameter tensor into the module's parameter data.
		/// </summary>
		/// <typeparam name="T">The numeric type of the module's parameters and tensors (e.g., float, double).</typeparam>
		/// <param name="model">Reference to the NN::Module<T> whose parameters will be populated from the checkpoint. The module's parameter count must match the serialized count.</param>
		/// <param name="reader">BinaryReader used to read the checkpoint data (parameter count followed by each parameter's tensor data).</param>
		template <typename T>
		static void LoadV1(NN::Module<T>& model, BinaryReader& reader);

		/// <summary>
		/// Serializes a module's named parameters into the provided BinaryWriter using the version 2 checkpoint format.
		/// </summary>
		/// <typeparam name="T">The element type of the module's tensors and parameters (e.g., float, double).</typeparam>
		/// <param name="model">The neural network module whose named parameters will be saved. The function calls model.GetNamedParameters() to obtain each parameter's name and tensor.</param>
		/// <param name="writer">The BinaryWriter used to write the serialized data: the parameter count, each parameter name length and name bytes, and each parameter's tensor via writer.WriteTensor.</param>
		template <typename T>
		static void SaveV2(const NN::Module<T>& model, BinaryWriter& writer);

		/// <summary>
		/// Loads parameter tensors from a binary checkpoint into the given model, matching parameters by name.
		/// </summary>
		/// <typeparam name="T">The numeric element type of the tensors stored in the model's parameters (e.g., float, double).</typeparam>
		/// <param name="model">Reference to the model whose named parameters will be populated. The function iterates the model's named parameters and updates each parameter's tensor with data read from the reader.</param>
		/// <param name="reader">BinaryReader used to read the checkpoint data. The reader must provide the number of parameters, each parameter name (length + bytes), and the serialized tensor data for each parameter.</param>
		template <typename T>
		static void LoadV2(NN::Module<T>& model, BinaryReader& reader);

		/// <summary>
		/// Serializes the given model and optional optimizer, scheduler, and trainer state into the provided BinaryWriter using the checkpoint V3 format.
		/// </summary>
		/// <typeparam name="T">Element/data type used by the model, optimizer, scheduler, and trainer (for example float or double).</typeparam>
		/// <param name="model">The model whose parameters and training state are serialized.</param>
		/// <param name="writer">The BinaryWriter used to write checkpoint sections and data to the output stream.</param>
		/// <param name="opt">Optional pointer to the optimizer. If non-null, the optimizer's type name and internal state are written.</param>
		/// <param name="scheduler">Optional pointer to the learning-rate scheduler. If non-null, the scheduler's type name and internal state are written.</param>
		/// <param name="state">Optional pointer to the trainer state. If non-null, current epoch, global step, and best-metric information (and best metric value if present) are written.</param>
		template <typename T>
		static void SaveV3(const NN::Module<T>& model, BinaryWriter& writer, const Optimizers::Optimizer<T>* opt, const Schedulers::LRScheduler<T>* scheduler, const Training::TrainerState<T>* state);

		/// <summary>
		/// Loads a version 3 checkpoint into the provided model and optionally restores optimizer, scheduler, and trainer state from a BinaryReader. The function first delegates to LoadV2, reads a training flag to set the model mode, then iterates checkpoint sections until Section::End, validating types and restoring state as present.
		/// </summary>
		/// <typeparam name="T">Numeric/precision type used by the model, optimizer, scheduler, and trainer (e.g., float or double). All provided components must use a compatible T; optimizer/scheduler type names are validated against the checkpoint.</typeparam>
		/// <param name="model">Reference to the NN::Module<T> into which model parameters/state will be loaded. The model is set to train or evaluate mode depending on the saved training flag and is passed to the optimizer when restoring optimizer state.</param>
		/// <param name="reader">Reference to a BinaryReader used to read the checkpoint stream. Must be positioned at the V3 checkpoint contents (LoadV2 is called first).</param>
		/// <param name="opt">Pointer to an Optimizers::Optimizer<T>. If non-null and the checkpoint contains an optimizer section, the optimizer's TypeName is validated and opt->LoadState(reader, model) is invoked. If the checkpoint contains optimizer state but this pointer is null, the function throws std::runtime_error.</param>
		/// <param name="scheduler">Pointer to a Schedulers::LRScheduler<T>. If non-null and the checkpoint contains a scheduler section, the scheduler's TypeName is validated and scheduler->LoadState(reader) is invoked. If the checkpoint contains scheduler state but this pointer is null, the function throws std::runtime_error.</param>
		/// <param name="state">Pointer to a Training::TrainerState<T>. If non-null and the checkpoint contains a trainer section, the function reads currentEpoch, globalStep, hasBestMetric and, if present, bestValidationMetric into this object. If the checkpoint contains trainer state but this pointer is null, the function throws std::runtime_error.</param>
		template <typename T>
		static void LoadV3(NN::Module<T>& model, BinaryReader& reader, Optimizers::Optimizer<T>* opt, Schedulers::LRScheduler<T>* scheduler, Training::TrainerState<T>* state);

	private:
		static constexpr uint64_t MAGIC_NUMBER = 0x4D4C434F5245ULL; /// "MLCORE" in hexadecimal
		static constexpr uint32_t FORMAT_VERSION = 3; /// Most recent developed version
	};
}

#include "checkpoint.inl"