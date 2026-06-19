 /// tensorDataset.h
#pragma once
#include <mlCore/data/dataset.h>

namespace MLCore::Data {
	/// <summary>
	/// A dataset wrapper that holds input and target tensors and provides indexed access to paired samples.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensors (for example float, double, or int).</typeparam>
	template <typename T>
	class TensorDataset : public Dataset<T> {
	public:
		/// <summary>
		/// Constructs a TensorDataset<T> from input and target tensors and verifies they contain the same number of samples. Throws std::runtime_error if the sample counts differ.
		/// </summary>
		/// <typeparam name="T">The element type stored in the tensors (for example, float or double).</typeparam>
		/// <param name="inputs">Tensor of input samples. Its first dimension is treated as the sample count and must match the targets tensor.</param>
		/// <param name="targets">Tensor of target samples. Its first dimension is treated as the sample count and must match the inputs tensor.</param>
		TensorDataset(const TensorCore::Tensor<T>& inputs, const TensorCore::Tensor<T>& targets);

		/// <summary>
		/// Returns the number of entries in the dataset (the size of the first dimension of m_Inputs).
		/// </summary>
		/// <typeparam name="T">The element type stored by the TensorDataset class template.</typeparam>
		/// <returns>The dataset size as a size_t, taken from m_Inputs.Dims()[0].</returns>
		virtual size_t Size() const override;

		/// <summary>
		/// Returns the item at the specified index as a pair of tensors.
		/// </summary>
		/// <param name="index">Zero-based index of the item to retrieve.</param>
		/// <returns>A std::pair containing two TensorCore::Tensor<T> objects associated with the item at the given index.</returns>
		virtual std::pair<TensorCore::Tensor<T>, TensorCore::Tensor<T>> GetItem(size_t index) const override;

	private:
		TensorCore::Tensor<T> m_Inputs; /// The input tensor for the containing class.
		TensorCore::Tensor<T> m_Targets; /// The target tenor for the containing class.
	};
}

#include "tensorDataset.inl"