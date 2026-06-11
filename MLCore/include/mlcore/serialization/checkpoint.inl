// checkpoint.inl
#include <filesystem>
#include <unordered_map>
#include <mlCore/tensor/dataType.h>

namespace MLCore::Serialization {
	template <typename T>
	void Checkpoint::Save(const NN::Module<T>& model, const std::string& path) {
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

		default:
			throw std::runtime_error("ERROR: Save: Support for this version doesn't exist");
		}

		
	}

	template <typename T>
	void Checkpoint::Load(NN::Module<T>& model, const std::string& path) {
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

			size_t numElements = tensor.NumElements();
			size_t rank = tensor.Rank();
			auto& dims = tensor.Dims();

			writer.Write(numElements);
			writer.Write(rank);
			writer.WriteArray(dims.data(), rank);
			writer.WriteArray(tensor.Data(), numElements);
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

			// Read the size of each parameter
			size_t numElements;
			reader.Read(numElements);

			if (numElements != tensor.NumElements()) {
				throw std::runtime_error("ERROR: Load: Checkpoint tensor size mismatch");
			}

			// Read the rank of each parameter
			size_t rank;
			reader.Read(rank);

			if (rank != tensor.Rank()) {
				throw std::runtime_error("ERROR: Load: Checkpoint tensor rank mismatch");
			}

			// Read the shape of each parameter
			std::vector<size_t> dims( rank );
			reader.ReadArray(dims.data(), dims.size());

			if (dims != tensor.Dims()) {
				throw std::runtime_error("ERROR: Load: Checkpoint tensor shape mismatch");
			}

			// Read the data of each parameter
			reader.ReadArray(tensor.Data(), numElements);
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

			size_t numElements = tensor.NumElements();
			size_t rank = tensor.Rank();
			auto& dims = tensor.Dims();

			writer.Write(numElements);
			writer.Write(rank);
			writer.WriteArray(dims.data(), rank);
			writer.WriteArray(tensor.Data(), numElements);
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

			// Read the size of each parameter
			size_t numElements;
			reader.Read(numElements);

			if (numElements != tensor.NumElements()) {
				throw std::runtime_error("ERROR: Load: Checkpoint tensor size mismatch");
			}

			// Read the rank of each parameter
			size_t rank;
			reader.Read(rank);

			if (rank != tensor.Rank()) {
				throw std::runtime_error("ERROR: Load: Checkpoint tensor rank mismatch");
			}

			// Read the shape of each parameter
			std::vector<size_t> dims( rank );
			reader.ReadArray(dims.data(), rank);

			if (dims != tensor.Dims()) {
				throw std::runtime_error("ERROR: Load: Checkpoint tensor shape mismatch");
			}

			// Read the data of each parameter
			reader.ReadArray(tensor.Data(), numElements);
		}
	}
}