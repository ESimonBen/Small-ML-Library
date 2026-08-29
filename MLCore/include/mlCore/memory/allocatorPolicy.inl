/// allocatorPolicy.inl

namespace MLCore::Memory {
	namespace Detail {
		template <typename Iterator>
		inline ArenaAllocator* ResolveFromAllocators(Iterator begin, Iterator end) {
			ArenaAllocator& sharedAllocator = Runtime::MLContext::GetSharedAllocator();
			ArenaAllocator* resolved = nullptr;

			for (auto it = begin; it < end; ++it) {
				ArenaAllocator* a = *it;

				if (!a || a == &sharedAllocator) {
					continue;
				}

				if (!resolved) {
					resolved = a;
				}
				else if (resolved != a) {
					return nullptr;
				}
			}

			return resolved ? resolved : &Runtime::MLContext::GetAllocator();
		}
	}

	template <typename T>
	inline ArenaAllocator* ResolveOperationAllocator(const TensorCore::Tensor<T>& A, const TensorCore::Tensor<T>& B) {
		ArenaAllocator& sharedArena = Runtime::MLContext::GetSharedAllocator();
		ArenaAllocator* a = &(A.GetAllocator());
		ArenaAllocator* b = &(B.GetAllocator());

		if (a == b) {
			return a;
		}

		if (a == &sharedArena && b != &sharedArena) {
			return b;
		}

		if (b == &sharedArena && a != &sharedArena) {
			return a;
		}

		return nullptr;
	}
	
	template<typename T>
	inline ArenaAllocator* ResolveOperationAllocator(std::initializer_list<const TensorCore::Tensor<T>*> tensors) {
		std::vector<ArenaAllocator*> allocators;
		allocators.reserve(tensors.size());

		for (const TensorCore::Tensor<T>* tensor : tensors) {
			allocators.push_back(tensor ? &tensor->GetAllocator() : nullptr);
		}

		return Detail::ResolveFromAllocators(allocators.begin(), allocators.end());
	}
	
	template<typename T>
	ArenaAllocator* ResolveOperationAllocator(const std::vector<TensorCore::Tensor<T>>& tensors) {
		std::vector<ArenaAllocator*> allocators;
		allocators.reserve(tensors.size());

		for (const TensorCore::Tensor<T>& tensor : tensors) {
			allocators.push_back(&tensor.GetAllocator());
		}

		return Detail::ResolveFromAllocators(allocators.begin(), allocators.end());
	}
}