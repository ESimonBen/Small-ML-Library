// checkpoint.inl
#include <filesystem>

namespace MLCore::Serialization {
	template <typename T>
	static void Checkpoint::Write(std::ofstream& out, const T& data) {
		out.write(reinterpret_cast<const char*>(&data), sizeof(data));

		if (!out) {
			throw std::runtime_error("ERROR: Write: Checkpoint write failed");
		}
	}

	template <typename T>
	static void Checkpoint::Read(std::ifstream& in, T& data) {
		if (!in.read(reinterpret_cast<char*>(&data), sizeof(data))) {
			throw std::runtime_error("ERROR: Load: Checkpoint read failed");
		}
	}

	template <typename T>
	static void Checkpoint::WriteArray(std::ofstream& out, const T* data, size_t count) {
		out.write(reinterpret_cast<const char*>(data), sizeof(T) * count);

		if (!out) {
			throw std::runtime_error("ERROR: Write: Checkpoint write failed");
		}
	}

	template <typename T>
	static void Checkpoint::ReadArray(std::ofstream& in, T* data, size_t count) {
		if (!in.read(reinterpret_cast<char*>(data), sizeof(T) * count)) {
			throw std::runtime_error("ERROR: Load: Checkpoint read failed");
		}
	}

	template <typename T>
	void Checkpoint::Save(const NN::Module<T>& model, const std::string& path) {
		// Create the file and the directories it's stored in
		std::filesystem::path filePath = path;
		std::filesystem::create_directories(filePath.parent_path());

		std::ofstream out{ filePath, std::ios::binary };

		if (!out) {
			throw std::runtime_error("ERROR: Save: Failed to open checkpoint path");
		}

		Write(out, MAGIC_NUMBER);
		Write(out, FORMAT_VERSION);

		switch (FORMAT_VERSION) {
		case 1:
			SaveV1(model, out);
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

		Read(in, magic);
		Read(in, version);

		if (magic != MAGIC_NUMBER) {
			throw std::runtime_error("ERROR: Load: Invalid checkpoint file");
		}

		switch (version) {
		case 1:
			LoadV1(model, in);
			break;

		default:
			throw std::runtime_error("ERROR: Load: Unsupported checkpoint version");
		}
	}

	template <typename T>
	void Checkpoint::SaveV1(const NN::Module<T>& model, std::ofstream& out) {
		// Store the number of parameters
		auto params = model.GetParameters();
		size_t numParams = params.size();

		Write(out, numParams);

		// Store the size, rank, shape, and data of each parameter
		for (auto& ref : params) {
			NN::Parameter<T>& param = ref.get();
			TensorCore::Tensor<T>& tensor = param.Data();

			size_t numElements = tensor.NumElements();
			size_t rank = tensor.Rank();
			auto& dims = tensor.Dims();

			Write(out, numElements);
			Write(out, rank);
			WriteArray(out, dims.data(), dims.size());
			WriteArray(out, tensor.Data(), numElements);
		}
	}

	template <typename T>
	void Checkpoint::LoadV1(NN::Module<T>& model, std::ifstream& in) {
		// Read the number of parameters
		auto params = model.GetParameters();
		size_t numParams;
		Read(in, numParams);

		if (numParams != params.size()) {
			throw std::runtime_error("ERROR: Load: Checkpoint parameter count mismatch");
		}

		// Read each parameter
		for (auto& ref : params) {
			NN::Parameter<T>& param = ref.get();
			TensorCore::Tensor<T>& tensor = param.Data();

			// Read the size of each parameter
			size_t numElements;
			Read(in, numElements);

			if (numElements != tensor.NumElements()) {
				throw std::runtime_error("ERROR: Load: Checkpoint tensor size mismatch");
			}

			// Read the rank of each parameter
			size_t rank;
			Read(in, rank);

			if (rank != tensor.Rank()) {
				throw std::runtime_error("ERROR: Load: Checkpoint tensor rank mismatch");
			}

			// Read the shape of each parameter
			std::vector<size_t> dims{ rank };
			ReadArray(in, dims.data(), dims.size());

			if (dims != tensor.Dims()) {
				throw std::runtime_error("ERROR: Load: Checkpoint tensor shape mismatch");
			}

			// Read the data of each parameter
			ReadArray(in, tensor.Data(), numElements);
		}
	}
}