/// operatorOverloads.h
#pragma once
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/elementwise/elementwise.h>

namespace MLCore::Operations {
	/// <summary>
	/// Returns the element-wise sum of two tensors.
	/// </summary>
	/// <typeparam name="T">Element type of the tensors.</typeparam>
	/// <param name="A">Left-hand tensor operand.</param>
	/// <param name="B">Right-hand tensor operand.</param>
	/// <returns>A TensorCore::Tensor containing the element-wise sum of A and B.</returns>
	template <typename T>
	TensorCore::Tensor<T> operator+(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B);

	/// <summary>
	/// Binary subtraction operator that returns the element-wise difference of two tensors.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensors.</typeparam>
	/// <param name="A">Left-hand operand tensor (minuend).</param>
	/// <param name="B">Right-hand operand tensor (subtrahend).</param>
	/// <returns>A TensorCore::Tensor containing the element-wise result of A - B.</returns>
	template <typename T>
	TensorCore::Tensor<T> operator-(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B);

	/// <summary>
	/// Overloaded multiplication operator that returns the product of two tensors by invoking Multiply.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensors.</typeparam>
	/// <param name="A">Left-hand tensor operand.</param>
	/// <param name="B">Right-hand tensor operand.</param>
	/// <returns>A TensorCore::Tensor containing the result of multiplying A and B.</returns>
	template <typename T>
	TensorCore::Tensor<T> operator*(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B);

	/// <summary>
	/// Returns the result of dividing one tensor by another using the library's Divide implementation.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensors.</typeparam>
	/// <param name="A">Left-hand operand (dividend) tensor.</param>
	/// <param name="B">Right-hand operand (divisor) tensor.</param>
	/// <returns>A TensorCore::Tensor containing the result of dividing A by B (typically element-wise), with the same shape or broadcasting semantics as defined by Divide.</returns>
	template <typename T>
	TensorCore::Tensor<T> operator/(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B);

	/// <summary>
	/// Returns a tensor obtained by adding a scalar to each element of the input tensor.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensor (e.g., float, double, int).</typeparam>
	/// <param name="Input">Constant reference to the input tensor whose elements will be incremented.</param>
	/// <param name="Scalar">The scalar value to add to each element of the input tensor.</param>
	/// <returns>A new TensorCore::Tensor containing the element-wise sum of Input and Scalar.</returns>
	template <typename T>
	TensorCore::Tensor<T> operator+(const TensorCore::Tensor<T>& Input, T Scalar);
	
	/// <summary>
	/// Adds a scalar to every element of a tensor (implements scalar + tensor).
	/// </summary>
	/// <typeparam name="T">The element type of the tensor and the scalar.</typeparam>
	/// <param name="Scalar">The scalar value to add to each element of the tensor.</param>
	/// <param name="Input">The input tensor (const reference); not modified.</param>
	/// <returns>A new TensorCore::Tensor containing the result (each element = Input element + Scalar).</returns>
	template <typename T>
	TensorCore::Tensor<T> operator+(T Scalar, const TensorCore::Tensor<T>& Input);

	/// <summary>
	/// Prefix increment operator that adds 1 to every element of the tensor.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor.</typeparam>
	/// <param name="Input">Reference to the tensor to increment; modified in-place by adding 1 to each element.</param>
	/// <returns>A tensor equal to Input after incrementing (returned by value).</returns>
	template <typename T>
	TensorCore::Tensor<T> operator++(TensorCore::Tensor<T>& Input);

	/// <summary>
	/// Postfix increment operator that adds 1 to every element of the tensor.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor.</typeparam>
	/// <param name="Input">Reference to the tensor to increment; modified in-place by adding 1 to each element.</param>
	/// <returns>A tensor equal to Input after incrementing (returned by value).</returns>
	template <typename T>
	TensorCore::Tensor<T> operator++(TensorCore::Tensor<T>& Input, int);

	/// <summary>
	/// Subtracts a scalar from each element of a tensor and returns the result.
	/// </summary>
	/// <typeparam name="T">The element type of the tensor and the scalar.</typeparam>
	/// <param name="Input">The input tensor whose elements will have the scalar subtracted.</param>
	/// <param name="Scalar">The scalar value to subtract from each element of the input tensor.</param>
	/// <returns>A new TensorCore::Tensor where each element equals the corresponding element of Input minus Scalar.</returns>
	template <typename T>
	TensorCore::Tensor<T> operator-(const TensorCore::Tensor<T>& Input, T Scalar);

	/// <summary>
	/// Computes a tensor where each element is the result of subtracting the input tensor element from the given scalar (i.e., Scalar - Input).
	/// </summary>
	/// <typeparam name="T">The type of the scalar and the tensor elements.</typeparam>
	/// <param name="Scalar">The scalar value to subtract the tensor elements from (the minuend).</param>
	/// <param name="Input">The tensor whose elements are subtracted from Scalar.</param>
	/// <returns>A TensorCore::Tensor containing the element-wise results of Scalar minus each element of Input.</returns>
	template <typename T>
	TensorCore::Tensor<T> operator-(T Scalar, const TensorCore::Tensor<T>& Input);

	/// <summary>
	/// Prefix decrement operator that subtracts one from every element of the tensor, modifying the input tensor in place.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor (numeric type) used for the decrement operation.</typeparam>
	/// <param name="Input">The tensor to decrement. This parameter is modified in place; each element is reduced by one.</param>
	/// <returns>The updated tensor after decrementing each element by one (returned by value).</returns>
	template <typename T>
	TensorCore::Tensor<T> operator--(TensorCore::Tensor<T>& Input);

	/// <summary>
	/// Postfix decrement operator that subtracts one from every element of the tensor, modifying the input tensor in place.
	/// </summary>
	/// <typeparam name="T">Element type of the tensor (numeric type) used for the decrement operation.</typeparam>
	/// <param name="Input">The tensor to decrement. This parameter is modified in place; each element is reduced by one.</param>
	/// <returns>The updated tensor after decrementing each element by one (returned by value).</returns>
	template <typename T>
	TensorCore::Tensor<T> operator--(TensorCore::Tensor<T>& Input, int);

	/// <summary>
	/// Scales a tensor by a scalar, returning a new tensor with each element multiplied by the scalar.
	/// </summary>
	/// <typeparam name="T">The element type of the tensor and the scalar; determines the numeric type used for multiplication.</typeparam>
	/// <param name="Input">The input tensor to be scaled.</param>
	/// <param name="Scalar">The scalar value to multiply each element of the input tensor by.</param>
	/// <returns>A new TensorCore::Tensor containing the result of element-wise multiplication of Input by Scalar.</returns>
	template <typename T>
	TensorCore::Tensor<T> operator*(const TensorCore::Tensor<T>& Input, T Scalar);

	/// <summary>
	/// Multiplies a tensor by a scalar and returns a new tensor with each element scaled by the scalar.
	/// </summary>
	/// <typeparam name="T">The element type of the tensor and scalar (e.g., float, double, int).</typeparam>
	/// <param name="Scalar">The scalar value to multiply each element of the input tensor.</param>
	/// <param name="Input">The input tensor to be scaled.</param>
	/// <returns>A TensorCore::Tensor containing the result of multiplying each element of Input by Scalar.</returns>
	template <typename T>
	TensorCore::Tensor<T> operator*(T Scalar, const TensorCore::Tensor<T>& Input);

	/// <summary>
	/// Performs element-wise division of a tensor by a scalar and returns the result as a new tensor.
	/// </summary>
	/// <typeparam name="T">The data type of the elements stored in the tensor.</typeparam>
	/// <param name="Input">The input tensor to be divided (passed by const reference).</param>
	/// <param name="Scalar">The scalar value to divide each element of the tensor by.</param>
	/// <returns>A new TensorCore::Tensor containing the result of dividing each element of Input by Scalar.</returns>
	template <typename T>
	TensorCore::Tensor<T> operator/(const TensorCore::Tensor<T>& Input, T Scalar);

	/// <summary>
	/// Overloads the division operator to perform element-wise division of a scalar by a tensor (scalar on the left).
	/// </summary>
	/// <typeparam name="T">The numeric type of the tensor elements and the scalar (e.g., float, double, int).</typeparam>
	/// <param name="Scalar">The scalar numerator (left-hand operand) to divide by each element of the tensor.</param>
	/// <param name="Input">The tensor (right-hand operand) whose elements serve as denominators; division is applied element-wise.</param>
	/// <returns>A TensorCore::Tensor containing the result of Scalar divided by each corresponding element of Input.</returns>
	template <typename T>
	TensorCore::Tensor<T> operator/(T Scalar, const TensorCore::Tensor<T>& Input);

	/// <summary>
	/// Performs an element-wise equality comparison between two tensors.
	/// </summary>
	/// <typeparam name="T">The element type of the tensors.</typeparam>
	/// <param name="A">The first tensor to compare.</param>
	/// <param name="B">The second tensor to compare.</param>
	/// <returns>A tensor containing the result of element-wise equality comparisons; each element indicates whether the corresponding elements of A and B are equal (typically represented as a boolean or equivalent value).</returns>
	template <typename T>
	TensorCore::Tensor<T> operator==(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B);

	/// <summary>
	/// Overloaded inequality operator that performs an element-wise comparison of two Tensor objects and returns a tensor with the comparison results.
	/// </summary>
	/// <typeparam name="T">Type of elements stored in the tensors.</typeparam>
	/// <param name="A">Left-hand tensor operand to compare. Must have the same shape and element type as B.</param>
	/// <param name="B">Right-hand tensor operand to compare. Must have the same shape and element type as A.</param>
	/// <returns>A Tensor containing the element-wise results of A != B (typically representing boolean-like values indicating inequality).</returns>
	template <typename T>
	TensorCore::Tensor<T> operator!=(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B);

	/// <summary>
	/// Performs an element-wise greater-than comparison between two tensors.
	/// </summary>
	/// <typeparam name="T">The element type of the tensors.</typeparam>
	/// <param name="A">The left-hand tensor operand.</param>
	/// <param name="B">The right-hand tensor operand.</param>
	/// <returns>A tensor containing the results of the element-wise comparison (typically a boolean mask or a tensor with values indicating where A > B).</returns>
	template <typename T>
	TensorCore::Tensor<T> operator>(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B);

	/// <summary>
	/// Performs an element-wise less-than comparison between two tensors.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensors.</typeparam>
	/// <param name="A">The left-hand operand tensor (const reference) whose elements are compared.</param>
	/// <param name="B">The right-hand operand tensor (const reference) whose elements are compared.</param>
	/// <returns>A TensorCore::Tensor whose elements represent the result of the element-wise comparison (for each position, whether A's element is less than B's element).</returns>
	template <typename T>
	TensorCore::Tensor<T> operator<(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B);

	/// <summary>
	/// Performs an element-wise greater-than-or-equal comparison between two tensors.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensors.</typeparam>
	/// <param name="A">The left-hand tensor operand.</param>
	/// <param name="B">The right-hand tensor operand.</param>
	/// <returns>A TensorCore::Tensor containing the element-wise results of the comparison (for each element, indicates whether the corresponding value in A is greater than or equal to that in B).</returns>
	template<typename T>
	TensorCore::Tensor<T> operator>=(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B);

	/// <summary>
	/// Performs an element-wise less-than-or-equal comparison between two tensors.
	/// </summary>
	/// <typeparam name="T">The element type of the input tensors.</typeparam>
	/// <param name="A">Left-hand operand tensor whose elements are compared to the corresponding elements of B.</param>
	/// <param name="B">Right-hand operand tensor whose elements are compared to the corresponding elements of A.</param>
	/// <returns>A tensor containing the element-wise comparison results (for each position, indicates whether the element in A is less than or equal to the element in B).</returns>
	template<typename T>
	TensorCore::Tensor<T> operator<=(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B);
}

#include "operatorOverloads.inl"