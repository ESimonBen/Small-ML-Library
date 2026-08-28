 /// binaryArchive.inl

namespace MLCore::Serialization {
	inline BinaryWriter::BinaryWriter(std::ofstream& out)
		: m_Out(out)
	{}
	
	template <typename T>
	inline void BinaryWriter::Write(const T& data) {
		static_assert(std::is_trivially_copyable_v<T>, "ERROR: BinaryWriter::Write requires a trivially copyable type");

		m_Out.write(reinterpret_cast<const char*>(&data), sizeof(data));

		if (!m_Out) {
			throw std::runtime_error("ERROR: Write: Checkpoint write failed");
		}
	}
	
	template <typename T>
	inline void BinaryWriter::WriteArray(const T* data, size_t count) {
		static_assert(std::is_trivially_copyable_v<T>, "ERROR: BinaryWriter::WriteArray requires a trivially copyable type");

		m_Out.write(reinterpret_cast<const char*>(data), sizeof(T) * count);

		if (!m_Out) {
			throw std::runtime_error("ERROR: Write: Checkpoint write failed");
		}
	}
	
	template <typename T>
	inline void BinaryWriter::WriteTensor(const TensorCore::Tensor<T>& tensor) {
		if (!tensor.IsContiguous()) {
			throw std::runtime_error("ERROR: WriteTensor: Cannot serialize a non-contiguous tensor");
		}

		size_t numElements = tensor.NumElements();
		size_t rank = tensor.Rank();
		auto& dims = tensor.Dims();

		Write(numElements);
		Write(rank);
		WriteArray(dims.data(), rank);
		WriteArray(tensor.Data(), numElements);
	}
	
	inline BinaryReader::BinaryReader(std::ifstream& in)
		: m_In(in)
	{}
	
	template <typename T>
	inline void BinaryReader::Read(T& data) {
		static_assert(std::is_trivially_copyable_v<T>, "ERROR: BinaryRead::Read requires a trivially copyable type");

		if (!m_In.read(reinterpret_cast<char*>(&data), sizeof(data))) {
			throw std::runtime_error("ERROR: Load: Checkpoint read failed");
		}
	}
	
	template <typename T>
	inline void BinaryReader::ReadArray(T* data, size_t count) {
		static_assert(std::is_trivially_copyable_v<T>, "ERROR: BinaryRead::ReadArray requires a trivially copyable type");

		if (!m_In.read(reinterpret_cast<char*>(data), sizeof(T) * count)) {
			throw std::runtime_error("ERROR: Load: Checkpoint read failed");
		}
	}
	
	template <typename T>
	inline void BinaryReader::ReadTensor(TensorCore::Tensor<T>& tensor) {
		if (!tensor.IsContiguous()) {
			throw std::runtime_error("ERROR: ReadTensor: Cannot load into a non-contiguous tensor");
		}

		/// Read the size of each parameter
		size_t numElements;
		Read(numElements);

		if (numElements != tensor.NumElements()) {
			throw std::runtime_error("ERROR: Checkpoint tensor size mismatch");
		}

		/// Read the rank of each parameter
		size_t rank;
		Read(rank);

		if (rank != tensor.Rank()) {
			throw std::runtime_error("ERROR: Checkpoint tensor rank mismatch");
		}

		/// Read the shape of each parameter
		std::vector<size_t> dims(rank);
		ReadArray(dims.data(), dims.size());

		if (dims != tensor.Dims()) {
			throw std::runtime_error("ERROR: Checkpoint tensor shape mismatch");
		}

		/// Read the data of each parameter
		ReadArray(tensor.Data(), numElements);
	}
}