// shape.inl

namespace MLCore::Utils {
	template <typename... Dimensions, typename>
	inline Shape::Shape(Dimensions... dims) 
		: m_Dims(dims) {

	}

	inline size_t Shape::Dim() const {
		return m_Dims.size();
	}

	inline size_t Shape::NumElements() const {
		return std::accumulate(m_Dims.begin(), m_Dims.end(), size_t(1), std::multiplies<size_t>()); // Takes all the elements of m_Dims and multiplies them together
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