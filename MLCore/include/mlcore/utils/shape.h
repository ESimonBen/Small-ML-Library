// shape.h
#pragma once
#include <vector>
#include <cstddef>
#include <type_traits>

namespace MLCore::Utils{
	class Shape {
	public:
		Shape() = default;
		Shape(const Shape& other) noexcept;
		Shape(Shape&& other) noexcept;
		Shape& operator=(Shape&& other) noexcept;

		explicit Shape(const std::vector<size_t>& dims);

		// Shape with any number of arguments (makes sure to have only integral types for the arguments)
		template <typename... Dimensions, typename = std::enable_if_t<(std::is_integral_v<Dimensions> && ...)>>
		explicit Shape(Dimensions... dims);

		size_t Rank() const;
		
		size_t NumElements() const;

		const std::vector<size_t>& Strides() const;

		size_t FlattenIndex(const std::vector<size_t>& indices) const;

		std::vector<size_t> UnflattenIndex(size_t index) const;

		const std::vector<size_t>& Dims() const;

		bool operator==(const Shape& other) const;

		bool operator!=(const Shape& other) const;

		size_t operator[](size_t i) const;

		// Possible function to be added
		/*static Shape Broadcast(const Shape& a, const Shape& b);*/

	private:
		void ComputeStrides();

	private:
		std::vector<size_t> m_Dims;
		std::vector<size_t> m_Strides;
		size_t m_NumElements = 0;
	};
}

#include "shape.inl"