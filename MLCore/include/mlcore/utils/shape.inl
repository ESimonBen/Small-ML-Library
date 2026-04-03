// shape.inl
#include <numeric>

namespace MLCore::Utils {
	template <typename... Dimensions, typename>
	inline Shape::Shape(Dimensions... dims) 
		: m_Dims{ static_cast<size_t>(dims)... } {
		ComputeStrides();
		m_NumElements = 1;
		for (size_t d : m_Dims) {
			m_NumElements *= d;
		}
	}

	inline size_t Shape::Dim() const {
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
		return m_Dims == other.m_Dims;
	}

	inline bool Shape::operator!=(const Shape& other) const {
		return !(*this == other);
	}
}