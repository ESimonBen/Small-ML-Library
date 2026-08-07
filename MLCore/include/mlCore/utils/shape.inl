 /// shape.inl
#include <numeric>
#include <stdexcept>

namespace MLCore::Utils {
	inline Shape::Shape(const std::vector<size_t>& dims)
		: m_Dims(dims) {
		ComputeStrides();

		m_NumElements = 1;
		for (size_t d : m_Dims) {
			m_NumElements *= d;
		}
	}
	
	template <typename... Dimensions, typename>
	inline Shape::Shape(Dimensions... dims) 
		: m_Dims{ static_cast<size_t>(dims)... } {
		ComputeStrides();
		m_NumElements = 1;
		for (size_t d : m_Dims) {
			m_NumElements *= d;
		}
	}
	
	inline Shape::Shape(const Shape& other) noexcept
		: m_Dims(other.m_Dims), m_Strides(other.m_Strides), m_NumElements(other.m_NumElements)
	{
	}
	
	inline Shape::Shape(Shape&& other) noexcept
		: m_Dims(std::move(other.m_Dims)), m_Strides(std::move(other.m_Strides)), m_NumElements(other.m_NumElements)
	{
	}
	
	inline Shape& Shape::operator=(const Shape& other) noexcept {
		if (*this != other) {
			m_Dims = other.m_Dims;
			m_Strides = other.m_Strides;
			m_NumElements = other.m_NumElements;
		}

		return *this;
	}

	inline Shape& Shape::operator=(Shape&& other) noexcept {
		if (*this != other) {
			m_Dims = std::move(other.m_Dims);
			m_Strides = std::move(other.m_Strides);
			m_NumElements = other.m_NumElements;
		}

		return *this;
	}
	
	inline size_t Shape::FlattenIndex(const std::vector<size_t>& indices) const {
		if (indices.size() != m_Dims.size()) {
			throw std::runtime_error("ERROR: Index dimension mismatch");
		}

		if (m_Strides.size() != m_Dims.size()) {
			throw std::runtime_error("ERROR: Shape strides not initialized");
		}

		size_t offset = 0;

		for (size_t i = 0; i < m_Dims.size(); ++i) {
			if (indices[i] >= m_Dims[i]) {
				throw std::out_of_range("ERROR: Index out of bounds");
			}

			offset += indices[i] * m_Strides[i];
		}

		return offset;
	}
	
	inline std::vector<size_t> Shape::UnflattenIndex(size_t index) const {
		std::vector<size_t> indices(m_Dims.size());

		for (size_t i = 0; i < m_Dims.size(); ++i) {
			indices[i] = index / m_Strides[i];
			index %= m_Strides[i];
		}

		return indices;
	}
	
	inline void Shape::ComputeStrides() {
		m_Strides.resize(m_Dims.size());

		if (!m_Dims.empty()) {
			m_Strides.back() = 1;
			for (int i = (int)m_Dims.size() - 2; i >= 0; --i) {
				m_Strides[i] = m_Strides[i + 1] * m_Dims[i + 1];
			}
		}
	}
	
	inline size_t Shape::operator[](size_t i) const {
		return m_Dims[i];
	}
	
	inline size_t Shape::Rank() const {
		return m_Dims.size();
	}
	
	inline size_t Shape::NumElements() const {
		if (m_Dims.empty()) {
			return 0;
		}

		return m_NumElements;
	}
	
	inline const std::vector<size_t>& Shape::Strides() const {
		return m_Strides;
	}
	
	inline const std::vector<size_t>& Shape::Dims() const {
		return m_Dims;
	}
	
	inline bool Shape::operator==(const Shape& other) const {
		bool equalSize = m_NumElements == other.m_NumElements;

		if (equalSize && m_NumElements == 0) {
			return true;
		}
		else if (equalSize) {
			return m_Dims == other.m_Dims;
		}

		return false;
	}
	
	inline bool Shape::operator!=(const Shape& other) const {
		return !(*this == other);
	}
}