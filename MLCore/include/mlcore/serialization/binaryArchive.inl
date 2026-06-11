// binaryArchive.inl

namespace MLCore::Serialization {
	BinaryWriter::BinaryWriter(std::ofstream& out)
		: m_Out(out)
	{}

	template <typename T>
	void BinaryWriter::Write(const T& data) {
		m_Out.write(reinterpret_cast<const char*>(&data), sizeof(data));

		if (!m_Out) {
			throw std::runtime_error("ERROR: Write: Checkpoint write failed");
		}
	}

	template <typename T>
	void BinaryWriter::WriteArray(const T* data, size_t count) {
		m_Out.write(reinterpret_cast<const char*>(data), sizeof(T) * count);

		if (!m_Out) {
			throw std::runtime_error("ERROR: Write: Checkpoint write failed");
		}
	}

	BinaryReader::BinaryReader(std::ifstream& in)
		: m_In(in)
	{}

	template <typename T>
	void BinaryReader::Read(T& data) {
		if (!m_In.read(reinterpret_cast<char*>(&data), sizeof(data))) {
			throw std::runtime_error("ERROR: Load: Checkpoint read failed");
		}
	}

	template <typename T>
	void BinaryReader::ReadArray(T* data, size_t count) {
		if (!m_In.read(reinterpret_cast<char*>(data), sizeof(T) * count)) {
			throw std::runtime_error("ERROR: Load: Checkpoint read failed");
		}
	}
}