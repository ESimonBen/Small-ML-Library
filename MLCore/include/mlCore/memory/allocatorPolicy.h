/// allocatorPolicy.h
#pragma once
#include <mlCore/tensor/tensor.h>

namespace MLCore::Memory {
	namespace Detail {
		/// <summary>
		/// Attempts to determine a single allocator from a range of allocator pointers. If all non-null, non-shared allocators in the range are the same, that allocator is returned; if none are found the shared allocator is returned; if conflicting allocators are found, nullptr is returned.
		/// </summary>
		/// <typeparam name="Iterator">Iterator type that dereferences to an ArenaAllocator* (or equivalent). The iterator must support increment, dereference, and range comparison (used with less-than symbol in the implementation).</typeparam>
		/// <param name="begin">Iterator to the first element in the range of allocator pointers.</param>
		/// <param name="end">Iterator one past the last element in the range.</param>
		/// <returns>Pointer to the resolved ArenaAllocator if a single non-shared allocator is found, a pointer to the shared allocator if no non-shared allocator is present, or nullptr if multiple different non-shared allocators are encountered.</returns>
		template <typename Iterator>
		ArenaAllocator* ResolveFromAllocators(Iterator begin, Iterator end);
	}

	/// <summary>
	/// Resolves which ArenaAllocator should be used for an operation involving two tensors.
	/// </summary>
	/// <typeparam name="T">Element type of the input tensors (type parameter of Tensor); not used in the allocator resolution logic.</typeparam>
	/// <param name="A">The first tensor whose allocator is considered.</param>
	/// <param name="B">The second tensor whose allocator is considered.</param>
	/// <returns>Pointer to the chosen ArenaAllocator to use for the operation, or nullptr if no single allocator can be selected. Behavior: if both tensors share the same allocator that allocator is returned; if one tensor uses the global shared allocator (Runtime::MLContext::GetSharedAllocator()) and the other uses a different allocator, the non-shared allocator is returned; otherwise nullptr is returned.</returns>
	template <typename T>
	ArenaAllocator* ResolveOperationAllocator(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B);

	/// <summary>
	/// Resolves an ArenaAllocator from a list of tensors by collecting each tensor's allocator (if the tensor is non-null) and delegating resolution to Detail::ResolveFromAllocators.
	/// </summary>
	/// <typeparam name="T">The element type of the tensors.</typeparam>
	/// <param name="tensors">An initializer_list of pointers to const Tensor. Each element may be nullptr; non-null tensors contribute their allocator to the resolution process.</param>
	/// <returns>A pointer to the resolved ArenaAllocator (not owned by the caller). May be nullptr if no allocator could be resolved.</returns>
	template <typename T>
	ArenaAllocator* ResolveOperationAllocator(std::initializer_list<const TensorCore::Tensor<T>*> tensors);

	/// <summary>
	/// Collects the underlying allocators from the provided tensors and resolves a single ArenaAllocator to use for operations involving those tensors.
	/// </summary>
	/// <typeparam name="T">The element type stored in the tensors.</typeparam>
	/// <param name="tensors">A vector of TensorCore::Tensor objects whose allocators will be collected. The function reads each tensor's allocator to determine a common operation allocator.</param>
	/// <returns>A pointer to an ArenaAllocator resolved by Detail::ResolveFromAllocators from the collected allocators. May be nullptr if no suitable allocator is found.</returns>
	template <typename T>
	ArenaAllocator* ResolveOperationAllocator(const std::vector<TensorCore::Tensor<T>>& tensors);
}

#include "allocatorPolicy.inl"