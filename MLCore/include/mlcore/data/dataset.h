 /// dataset.h
#pragma once
#include <utility>
#include <cstddef>
#include <mlCore/tensor/tensor.h>

namespace MLCore::Data {
	/// <summary>
	/// Abstract generic dataset interface that provides the number of items and access to individual items as tensors.
	/// </summary>
	/// <typeparam name="T">Element type stored in the dataset and used by the returned tensors.</typeparam>
	template <typename T>
	class Dataset {
	public:
		/// <summary>
		/// Defaulted virtual destructor for Dataset that ensures derived objects are properly destroyed when deleted through a base pointer.
		/// </summary>
		virtual ~Dataset() = default;

		/// <summary>
		/// Pure virtual member function that returns the size (number of elements) of the object. Must be overridden by derived classes.
		/// </summary>
		/// <returns>The size of the object (number of elements) as a value of type size_t.</returns>
		virtual size_t Size() const = 0;

		/// <summary>
		/// Retrieves the pair of tensors associated with the specified index.
		/// </summary>
		/// <param name="index">Zero-based index of the item to retrieve.</param>
		/// <returns>A std::pair containing two TensorCore::Tensor<T> instances corresponding to the item at the given index; the pair's first and second elements are the two tensors associated with that index.</returns>
		virtual std::pair<TensorCore::Tensor<T>, TensorCore::Tensor<T>> GetItem(size_t index) const = 0;
	};
}