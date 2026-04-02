#pragma once
#include <cstddef>
#include <mlcore/memory/allocator.h>

namespace mlcore::memory {
	template <typename T>
	class Storage {
	public:
		Storage(T* data, size_t size)
			: data(data), size(size) 
		{}

		T* data() {
			return m_Data;
		}

		const T* data() const {
			return m_Data;
		}

		size_t size() const {
			return m_Size;
		}

	private:
		T* m_Data;
		size_t m_Size; // Number of elements (NOT size in bytes)
	};

	template <typename T>
	inline Storage<T> MakeAllocation(ArenaAllocator& allocator, size_t size) {
		T* ptr = allocator.allocate<T>(size);
		return Storage<T>(ptr, size);
	}
}