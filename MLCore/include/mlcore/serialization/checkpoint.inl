 /// checkpoint.inl
#include <filesystem>
#include <unordered_map>
#include <mlCore/tensor/dataType.h>

namespace MLCore::Serialization {
	template <typename T>
	void Checkpoint::Save(const NN::Module<T>& model, const std::string& path, const Optimizers::Optimizer<T>* opt,
						  const Schedulers::LRScheduler<T>* scheduler, const Training::TrainerState<T>* state) {
		// Create the file and the directories it's stored in
		std::filesystem::path filePath = path;
		std::filesystem::path parentPath = filePath.parent_path();
		
		if (!parentPath.empty()) {
			std::filesystem::create_directories(parentPath);
		}

		std::ofstream out{ filePath, std::ios::binary };

		if (!out) {
			throw std::runtime_error("ERROR: Save: Failed to open checkpoint path");
		}

		BinaryWriter writer{ out };

		writer.Write(MAGIC_NUMBER);
		writer.Write(FORMAT_VERSION);

		TensorCore::DataType dtype = TensorCore::ExpectedType<T>();
		writer.Write(dtype);

		switch (FORMAT_VERSION) {
		case 1:
			SaveV1(model, writer);
			break;

		case 2:
			SaveV2(model, writer);
			break;

		case 3:
			SaveV3(model, writer, opt, scheduler, state);
			break;

		default:
			throw std::runtime_error("ERROR: Save: Support for this version doesn't exist");
		}

		
	}

	template <typename T>
	void Checkpoint::Load(NN::Module<T>& model, const std::string& path, Optimizers::Optimizer<T>* opt,
						  Schedulers::LRScheduler<T>* scheduler, Training::TrainerState<T>* state) {
		std::ifstream in{ path, std::ios::binary };

		if (!in) {
			throw std::runtime_error("ERROR: Load: Failed to open checkpoint path");
		}

		// Read the magic number and file format version
		uint64_t magic;
		uint32_t version;
		TensorCore::DataType dtype;

		BinaryReader reader{ in };

		reader.Read(magic);
		reader.Read(version);
		reader.Read(dtype);

		if (magic != MAGIC_NUMBER) {
			throw std::runtime_error("ERROR: Load: Invalid checkpoint file");
		}

		if (dtype != TensorCore::ExpectedType<T>()) {
			throw std::runtime_error("ERROR: Load: Model data type mismatch");
		}

		switch (version) {
		case 1:
			LoadV1(model, reader);
			break;

		case 2:
			LoadV2(model, reader);
			break;

		case 3:
			LoadV3(model, reader, opt, scheduler, state);
			break;

		default:
			throw std::runtime_error("ERROR: Load: Unsupported checkpoint version");
		}
	}

	template <typename T>
	void Checkpoint::SaveV1(const NN::Module<T>& model, BinaryWriter& writer) {
		// Store the number of parameters
		auto params = model.GetParameters();
		size_t numParams = params.size();

		writer.Write(numParams);

		// Store the size, rank, shape, and data of each parameter
		for (const auto& ref : params) {
			const NN::Parameter<T>& param = ref.get();
			const TensorCore::Tensor<T>& tensor = param.Data();
			writer.WriteTensor(tensor);
		}
	}

	template <typename T>
	void Checkpoint::LoadV1(NN::Module<T>& model, BinaryReader& reader) {
		// Read the number of parameters
		auto params = model.GetParameters();
		size_t numParams;
		reader.Read(numParams);

		if (numParams != params.size()) {
			throw std::runtime_error("ERROR: Load: Checkpoint parameter count mismatch");
		}

		// Read each parameter
		for (auto& ref : params) {
			NN::Parameter<T>& param = ref.get();
			TensorCore::Tensor<T>& tensor = param.Data();
			reader.ReadTensor(tensor);
		}
	}

	template <typename T>
	void Checkpoint::SaveV2(const NN::Module<T>& model, BinaryWriter& writer) {
		std::vector<NN::ConstNamedParameter<T>> params = model.GetNamedParameters();

		size_t count = params.size();
		writer.Write(count);

		for (const auto& [name, p] : params) {
			size_t nameLength = name.size();
			writer.Write(nameLength);
			writer.WriteArray(name.data(), nameLength);

			const TensorCore::Tensor<T>& tensor = p.get().Data();
			writer.WriteTensor(tensor);
		}
	}

	template <typename T>
	void Checkpoint::LoadV2(NN::Module<T>& model, BinaryReader& reader) {
		std::unordered_map<std::string, NN::Parameter<T>*> paramMap;
		std::vector<NN::NamedParameter<T>> params = model.GetNamedParameters();

		for (auto& [name, p] : params) {
			if (paramMap.contains(name)) {
				throw std::runtime_error("ERROR: Duplicate parameter name");
			}

			paramMap[name] = &(p.get());
		}

		size_t numParams;
		reader.Read(numParams);

		if (numParams != params.size()) {
			throw std::runtime_error("ERROR: Load: Checkpoint parameter count mismatch");
		}

		for (size_t i = 0; i < numParams; ++i) {
			size_t nameLength;
			reader.Read(nameLength);

			std::string name( nameLength, '\0' );
			reader.ReadArray(name.data(), nameLength);

			auto iter = paramMap.find(name);

			if (iter == paramMap.end()) {
				throw std::runtime_error("ERROR: Load: Checkpoint name mismatch");
			}

			TensorCore::Tensor<T>& tensor = iter->second->Data();
			reader.ReadTensor(tensor);
		}
	}

	template <typename T>
	void Checkpoint::SaveV3(const NN::Module<T>& model, BinaryWriter& writer, const Optimizers::Optimizer<T>* opt, const Schedulers::LRScheduler<T>* scheduler, const Training::TrainerState<T>* state) {
		SaveV2(model, writer);

		writer.Write(model.IsTraining());

		if (opt) {
			writer.Write(Section::Optimizer);
			size_t nameLength = opt->TypeName().size();
			writer.Write(nameLength);
			writer.WriteArray(opt->TypeName().data(), nameLength);
			opt->SaveState(writer, model);
		}

		if (scheduler) {
			writer.Write(Section::Scheduler);
			size_t nameLength = scheduler->TypeName().size();
			writer.Write(nameLength);
			writer.WriteArray(scheduler->TypeName().data(), nameLength);
			scheduler->SaveState(writer);
		}

		if (state) {
			writer.Write(Section::Trainer);
			writer.Write(state->currentEpoch);
			writer.Write(state->globalStep);
			writer.Write(state->hasBestMetric);

			if (state->hasBestMetric) {
				writer.Write(state->bestValidationMetric);
			}
		}

		writer.Write(Section::End);
	}

	template <typename T>
	void Checkpoint::LoadV3(NN::Module<T>& model, BinaryReader& reader, Optimizers::Optimizer<T>* opt, Schedulers::LRScheduler<T>* scheduler, Training::TrainerState<T>* state) {
		LoadV2(model, reader);

		bool training;
		reader.Read(training);

		if (training) {
			model.Train();
		}
		else {
			model.Evaluate();
		}

		// New way
		while (true) {
			Section section;
			reader.Read(section);

			if (section == Section::End) {
				break;
			}

			switch (section) {
			case Section::Optimizer:
				{
					if (!opt) {
						throw std::runtime_error("ERROR: Checkpoint contains optimizer state, but no optimizer was provided");
					}
					
					size_t nameLength;
					reader.Read(nameLength);

					std::string expected = opt->TypeName();
					std::string actual(nameLength, '\0');
					reader.ReadArray(actual.data(), nameLength);

					if (expected != actual) {
						throw std::runtime_error("ERROR: Optimizer type mismatch");
					}

					opt->LoadState(reader, model);
				}
				break;

			case Section::Scheduler:
				{
					if (!scheduler) {
						throw std::runtime_error("ERROR: Checkpoint contains scheduler state, but no scheduler was provided");
					}
					
					size_t nameLength;
					reader.Read(nameLength);

					std::string expected = scheduler->TypeName();
					std::string actual(nameLength, '\0');
					reader.ReadArray(actual.data(), nameLength);

					if (expected != actual) {
						throw std::runtime_error("ERROR: Scheduler type mismatch");
					}

					scheduler->LoadState(reader);
				}
				break;

			case Section::Trainer:
				{
					if (!state) {
						throw std::runtime_error("ERROR: Checkpoint contains trainer state, but no TrainerState was provided");
					}

					reader.Read(state->currentEpoch);
					reader.Read(state->globalStep);
					reader.Read(state->hasBestMetric);

					if (state->hasBestMetric) {
						reader.Read(state->bestValidationMetric);
					}
				}
				break;

			default:
				throw std::runtime_error("ERROR: Unknown checkpoint section");
			}
		}
	}
}