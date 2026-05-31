// checkpoint.inl
#include <fstream>
#include <filesystem>

namespace MLCore::Serialization {
	template <typename T>
	void Checkpoint::Save(const NN::Module<T>& model, const std::string& path) {
		// Create the file and the directories it's stored in
		std::filesystem::path filePath = path;
		std::filesystem::create_directories(filePath.parent_path());

		std::ofstream out{ filePath, std::ios::binary };

		if (!out) {
			throw std::runtime_error("ERROR: Save: Failed to open checkpoint path");
		}

		// Store magic number and file format version
		out.write(reinterpret_cast<const char*>(MAGIC_NUMBER), sizeof(MAGIC_NUMBER));
		out.write(reinterpret_cast<const char*>(FORMAT_VERSION), sizeof(FORMAT_VERSION));

		// Store the number of parameters
		auto params = model.GetParameters();
		size_t numParams = params.size();

		out.write(reinterpret_cast<const char*>(&numParams), sizeof(size_t));

		// Store the size, rank, shape, and data of each parameter
		for (auto& ref : params) {
			NN::Parameter<T>& param = ref.get();
			TensorCore::Tensor<T>& tensor = param.Data();

			size_t numElements = tensor.NumElements();
			size_t rank = tensor.Rank();
			auto& dims = tensor.Dims();

			out.write(reinterpret_cast<const char*>(&numElements), sizeof(size_t));
			out.write(reinterpret_cast<const char*>(&rank), sizeof(size_t));
			out.write(reinterpret_cast<const char*>(&dims), sizeof(size_t) * dims.size());
			out.write(reinterpret_cast<const char*>(tensor.Data()), sizeof(T) * numElements);
		}
	}

	template <typename T>
	void Checkpoint::Load(NN::Module<T>& model, const std::string& path) {
		std::ifstream in{ path, std::ios::binary };

		if (!in) {
			throw std::runtime_error("ERROR: Load: Failed to open checkpoint path");
		}

		// Read the magic number
		uint32_t magic;

		if (!in.read(reinterpret_cast<char*>(&magic), sizeof(magic))) {
			throw std::runtime_error("ERROR: Load: Checkpoint read failed");
		}

		if (magic != MAGIC_NUMBER) {
			throw std::runtime_error("ERROR: Load: Invalid checkpoint file");
		}

		// Read the file format version
		uint32_t version;

		if (!in.read(reinterpret_cast<char*>(&version), sizeof(version))) {
			throw std::runtime_error("ERROR: Load: Checkpoint read failed");
		}

		if (version != FORMAT_VERSION) {
			throw std::runtime_error("ERROR: Load: Unsupported checkpoint version");
		}

		// Read the number of parameters
		auto params = model.GetParameters();
		size_t numParams;

		if (!in.read(reinterpret_cast<char*>(&numParams), sizeof(size_t))) {
			throw std::runtime_error("ERROR: Load: Checkpoint read failed");
		}

		if (numParams != params.size()) {
			throw std::runtime_error("ERROR: Load: Checkpoint parameter count mismatch");
		}

		// Read each parameter
		for (auto& ref : params) {
			NN::Parameter<T>& param = ref.get();
			TensorCore::Tensor<T>& tensor = param.Data();

			// Read the size of each parameter
			size_t numElements;

			if (!in.read(reinterpret_cast<char*>(&numElements), sizeof(size_t))) {
				throw std::runtime_error("ERROR: Load: Checkpoint read failed");
			}

			if (numElements != tensor.NumElements()) {
				throw std::runtime_error("ERROR: Load: Checkpoint tensor size mismatch");
			}

			// Read the rank of each parameter
			size_t rank;

			if (!in.read(reinterpret_cast<char*>(&rank), sizeof(size_t))) {
				throw std::runtime_error("ERROR: Load: Checkpoint read failed");
			}

			if (rank != tensor.Rank()) {
				throw std::runtime_error("ERROR: Load: Checkpoint tensor rank mismatch");
			}

			// Read the shape of each parameter
			std::vector<size_t> dims{ rank };

			if (!in.read(reinterpret_cast<char*>(dims.data()), sizeof(size_t) * dims.size())) {
				throw std::runtime_error("ERROR: Load: Checkpoint read failed");
			}

			if (dims != tensor.Dims()) {
				throw std::runtime_error("ERROR: Load: Checkpoint tensor shape mismatch");
			}

			// Read the data of each parameter
			if (!in.read(reinterpret_cast<char*>(tensor.Data()), sizeof(T) * numElements)) {
				throw std::runtime_error("ERROR: Load: Checkpoint read failed");
			}
		}
	}
}