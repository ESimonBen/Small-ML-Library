// shape.cpp
#include <numeric>
#include <stdexcept>
#include <mlCore/utils/shape.h>

namespace MLCore::Utils {
	Shape::Shape(const std::vector<size_t>& dims) 
		: m_Dims(dims) {
		ComputeStrides();

		m_NumElements = 1;
		for (size_t d : m_Dims) {
			m_NumElements *= d;
		}
	}

	Shape::Shape(const Shape& other) noexcept
		: m_Dims(other.m_Dims), m_Strides(other.m_Strides), m_NumElements(other.m_NumElements)
	{}

	Shape::Shape(Shape&& other) noexcept
		: m_Dims(std::move(other.m_Dims)), m_Strides(std::move(other.m_Strides)), m_NumElements(other.m_NumElements)
	{}

	Shape& Shape::operator=(const Shape& other) noexcept {
		if (*this != other) {
			m_Dims = other.m_Dims;
			m_Strides = other.m_Strides;
			m_NumElements = other.m_NumElements;
		}

		return *this;
	}

	Shape& Shape::operator=(Shape&& other) noexcept {
		if (*this != other) {
			m_Dims = std::move(other.m_Dims);
			m_Strides = std::move(other.m_Strides);
			m_NumElements = other.m_NumElements;
		}

		return *this;
	}
	
	size_t Shape::FlattenIndex(const std::vector<size_t>& indices) const {
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
	
	std::vector<size_t> Shape::UnflattenIndex(size_t index) const {
		std::vector<size_t> indices(m_Dims.size());

		for (size_t i = 0; i < m_Dims.size(); ++i) {
			indices[i] = index / m_Strides[i];
			index %= m_Strides[i];
		}

		return indices;
	}

	void Shape::ComputeStrides() {
		m_Strides.resize(m_Dims.size());

		if (!m_Dims.empty()) {
			m_Strides.back() = 1;
			for (int i = (int)m_Dims.size() - 2; i >= 0; --i) {
				m_Strides[i] = m_Strides[i + 1] * m_Dims[i + 1];
			}
		}
	}

	size_t Shape::operator[](size_t i) const {
		return m_Dims[i];
	}
}