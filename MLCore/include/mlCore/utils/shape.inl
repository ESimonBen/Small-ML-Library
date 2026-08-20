 /// shape.inl
#include <numeric>
#include <stdexcept>

namespace MLCore::Utils {
	inline Shape::Shape(const std::vector<size_t>& dims)
		: m_Dims(dims) {
		if (m_Dims.empty()) {
			m_NumElements = 0;
			return;
		}

		m_NumElements = 1;
		for (size_t d : m_Dims) {
			m_NumElements *= d;
		}
	}
	
	template <typename... Dimensions, typename>
	inline Shape::Shape(Dimensions... dims) 
		: m_Dims{ static_cast<size_t>(dims)... } {
		if (m_Dims.empty()) {
			m_NumElements = 0;
			return;
		}

		m_NumElements = 1;
		for (size_t d : m_Dims) {
			m_NumElements *= d;
		}
	}
	
	inline Shape::Shape(const Shape& other) noexcept
		: m_Dims(other.m_Dims), m_NumElements(other.m_NumElements)
	{}
	
	inline Shape::Shape(Shape&& other) noexcept
		: m_Dims(std::move(other.m_Dims)), m_NumElements(other.m_NumElements)
	{}
	
	inline Shape& Shape::operator=(const Shape& other) noexcept {
		if (*this != other) {
			m_Dims = other.m_Dims;
			m_NumElements = other.m_NumElements;
		}

		return *this;
	}

	inline Shape& Shape::operator=(Shape&& other) noexcept {
		if (*this != other) {
			m_Dims = std::move(other.m_Dims);
			m_NumElements = other.m_NumElements;
		}

		return *this;
	}
	
	inline size_t Shape::operator[](size_t i) const {
		return m_Dims[i];
	}
	
	inline size_t Shape::Rank() const {
		return m_Dims.size();
	}
	
	inline size_t Shape::NumElements() const {
		return m_NumElements;
	}
	
	inline const std::vector<size_t>& Shape::Dims() const {
		return m_Dims;
	}
	
	inline bool Shape::operator==(const Shape& other) const {
		return m_Dims == other.m_Dims;
	}
	
	inline bool Shape::operator!=(const Shape& other) const {
		return !(*this == other);
	}
}