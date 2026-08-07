 /// shape.h
#pragma once
#include <vector>
#include <cstddef>
#include <type_traits>

namespace MLCore::Utils{
	/// <summary>
	/// Represents the shape of a multi-dimensional array or tensor, including its dimensions, strides, and total number of elements.
	/// </summary>
	class Shape {
	public:
		/// <summary>
		/// Defaulted constructor that initializes a new Shape instance.
		/// </summary>
		Shape() = default;

		/// <summary>
		/// Copy constructor. Initializes a new Shape by copying the dimensions, strides, and number of elements from another Shape. The operation is noexcept and does not throw.
		/// </summary>
		/// <param name="other">The Shape to copy from. Its m_Dims, m_Strides, and m_NumElements are copied into the new object.</param>
		Shape(const Shape& other) noexcept;

		/// <summary>
		/// Move constructor that initializes a new Shape by taking ownership of the resources from another Shape. It moves m_Dims and m_Strides and copies m_NumElements; the operation is noexcept.
		/// </summary>
		/// <param name="other">The Shape to move from. Its m_Dims and m_Strides are moved into the new instance and m_NumElements is copied. The source object is left in a valid but unspecified state.</param>
		Shape(Shape&& other) noexcept;

		/// <summary>
		/// Copy assignment operator for Shape. Copies dimensions, strides, and element count from another Shape if they differ; marked noexcept.
		/// </summary>
		/// <param name="other">The Shape to copy from. If this and other are equal, no members are modified.</param>
		/// <returns>A reference to this Shape (i.e., *this) after assignment.</returns>
		Shape& operator=(const Shape& other) noexcept;

		/// <summary>
		/// Move-assigns the contents of another Shape into this one. Transfers internal resources and updates element count; marked noexcept.
		/// </summary>
		/// <param name="other">An rvalue reference to the Shape to move from. The function moves m_Dims and m_Strides from this object and copies m_NumElements. The moved-from object is left in a valid but unspecified state. If *this compares equal to other, no action is taken.</param>
		/// <returns>A reference to this Shape after assignment.</returns>
		Shape& operator=(Shape&& other) noexcept;

		/// <summary>
		/// Constructs a Shape object from the provided dimensions, computes internal strides, and calculates the total number of elements.
		/// </summary>
		/// <param name="dims">A vector of sizes for each dimension. Used to initialize the shape's internal dimensions; ComputeStrides() is called and the total number of elements is set to the product of these sizes.</param>
		explicit Shape(const std::vector<size_t>& dims);

		/// <summary>
		/// Constructs a Shape from a pack of integral dimension sizes, initializing the internal dimensions vector, computing row-major strides, and calculating the total number of elements.
		/// </summary>
		/// <typeparam name="Dimensions">A parameter pack of integral types used to specify each dimension size; the constructor is enabled only when all types are integral.</typeparam>
		/// <param name="dims">A list of dimension sizes (one per axis), provided as individual integral arguments; each value is interpreted as the size for the corresponding dimension.</param>
		/// <remarks>
		/// The constructor is explicit to avoid implicit conversions from a sequence of integers; ComputeStrides() is invoked and m_NumElements is set to the product of the provided sizes.
		/// </remarks>
		template <typename... Dimensions, typename = std::enable_if_t<(std::is_integral_v<Dimensions> && ...)>>
		explicit Shape(Dimensions... dims);

		/// <summary>
		/// Returns the rank (number of dimensions) of the shape.
		/// </summary>
		/// <returns>The number of dimensions in the shape as a size_t (the value of m_Dims.size()).</returns>
		size_t Rank() const;
		
		/// <summary>
		/// Returns the total number of elements described by this shape.
		/// </summary>
		/// <returns>The total number of elements (size_t). Returns 0 if the shape has no dimensions.</returns>
		size_t NumElements() const;
		
		/// <summary>
		/// Returns the shape's strides as a constant reference.
		/// </summary>
		/// <returns>A const reference to the std::vector<size_t> that holds the strides for this Shape. The reference refers to the object's internal storage and remains valid as long as the Shape is not modified or destroyed.</returns>
		const std::vector<size_t>& Strides() const;

		/// <summary>
		/// Converts a multidimensional index into a flat linear offset using the shape's strides. Validates that the provided indices match the shape's dimensionality and that strides are initialized. Throws std::runtime_error if the index vector size or strides are inconsistent with the shape, and std::out_of_range if any index is outside its dimension's bounds.
		/// </summary>
		/// <param name="indices">A vector of size_t indices, one per dimension. Must have the same number of elements as the shape's dimensions and each entry must be less than the corresponding dimension size.</param>
		/// <returns>The flattened offset (linear index) as a size_t computed by summing indices[i] * strides[i].</returns>
		size_t FlattenIndex(const std::vector<size_t>& indices) const;

		/// <summary>
		/// Converts a flattened (linear) index into its multi-dimensional indices using the shape's strides.
		/// </summary>
		/// <param name="index">The zero-based flattened index to convert. For meaningful results, index should be less than the total number of elements represented by this Shape.</param>
		/// <returns>A std::vector<size_t> of length equal to the number of dimensions (m_Dims.size()), where each element is the index along that dimension. If index is within bounds, indices[i] will be in the range [0, m_Dims[i]).</returns>
		std::vector<size_t> UnflattenIndex(size_t index) const;
		
		/// <summary>
		/// Returns a const reference to the shape's dimension sizes.
		/// </summary>
		/// <returns>A const reference to the internal std::vector<size_t> that holds the dimension sizes. The reference remains valid as long as the Shape object exists and its internal vector is not modified.</returns>
		const std::vector<size_t>& Dims() const;

		/// <summary>
		/// Determines whether this Shape is equal to another by comparing their dimensions.
		/// </summary>
		/// <param name="other">The Shape to compare with this instance.</param>
		/// <returns>true if the two Shape objects have equal dimensions (m_Dims); otherwise false.</returns>
		bool operator==(const Shape& other) const;

		/// <summary>
		/// Determines whether this Shape is not equal to another by negating the equality operator.
		/// </summary>
		/// <param name="other">The Shape to compare against this instance.</param>
		/// <returns>true if this Shape is not equal to other; otherwise false.</returns>
		bool operator!=(const Shape& other) const;

		/// <summary>
		/// Returns the dimension value at the specified index.
		/// </summary>
		/// <param name="i">Index of the dimension to retrieve. Behavior is undefined if the index is out of range.</param>
		/// <returns>The dimension at the given index as size_t.</returns>
		size_t operator[](size_t i) const;

	private:
		/// <summary>
		/// Recomputes the shape's internal strides from its dimensions in row-major order.
		/// </summary>
		void ComputeStrides();

	private:
		std::vector<size_t> m_Dims; /// Member variable that holds the size of each dimension.
		std::vector<size_t> m_Strides; /// Container holding stride values, typically one entry per dimension or axis.
		size_t m_NumElements = 0; /// A size_t variable that stores the current number of elements; initialized to 0.
	};
}

#include "shape.inl"