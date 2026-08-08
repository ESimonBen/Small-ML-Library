 /// dataLoader.h
#pragma once
#include <vector>
#include <mlCore/data/dataset.h>

namespace MLCore::Data {
	/// <summary>
	/// A template data loader that iterates over a Dataset<T> in fixed-size batches, optionally shuffling the sample order.
	/// </summary>
	/// <typeparam name="T">The element type stored in the dataset and tensors (e.g., the numeric type of features/labels).</typeparam>
	template <typename T>
	class DataLoader {
	public:
		/// <summary>
		/// Constructs a DataLoader<T> initialized with the given dataset, batch size, and shuffle mode. Throws std::runtime_error if batchSize is zero.
		/// </summary>
		/// <typeparam name="T">Type of the elements/samples stored in the dataset.</typeparam>
		/// <param name="dataset">Const reference to the Dataset<T> to load data from.</param>
		/// <param name="batchSize">Number of items per batch; must be greater than 0.</param>
		/// <param name="shuffle">Whether to shuffle the dataset between epochs (true) or preserve order (false).</param>
		DataLoader(const Dataset<T>& dataset, size_t batchSize, bool shuffle = true);

		/// <summary>
		/// Resets the data loader to the start of the dataset and optionally reshuffles the sample indices.
		/// </summary>
		/// <typeparam name="T">The type of elements managed by the DataLoader (the dataset sample type).</typeparam>
		/// <param name="reshuffle">If true, repopulates and randomly shuffles the internal index order; if false, leaves indices in sequential order.</param>
		void Reset(bool reshuffle = true);

		/// <summary>
		/// Checks whether the DataLoader has more items available to load.
		/// </summary>
		/// <typeparam name="T">The type of elements managed by the DataLoader.</typeparam>
		/// <returns>true if there are more items to read (the current index is less than the number of indices); otherwise false.</returns>
		bool HasNext() const;

		/// <summary>
		/// Returns the next batch of input and target tensors from the dataset and advances the loader's current index. Throws std::out_of_range if no batches remain.
		/// </summary>
		/// <typeparam name="T">The element type of the tensors returned (for example float, double, int).</typeparam>
		/// <returns>A pair where the first element is a concatenated tensor of the batch inputs and the second element is a concatenated tensor of the batch targets.</returns>
		std::pair<TensorCore::Tensor<T>, TensorCore::Tensor<T>> Next();

	private:
		const Dataset<T>& m_Dataset; /// Const reference to the Dataset<T> to load data from.
		size_t m_BatchSize; /// Number of items per batch.
		bool m_Shuffle; /// Whether to shuffle the dataset between epochs (true) or preserve order (false).
		size_t m_CurrentIndex = 0; /// Member variable that stores the current index or position, initialized to 0.
		std::vector<size_t> m_Indices; /// A member variable that holds a sequence of index values.
	};
}

#include "dataLoader.inl"