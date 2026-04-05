// storage.h
#pragma once
#include <cstddef>
#include <mlcore/memory/allocator.h>

namespace MLCore::Memory {
	template <typename T>
	class Storage {
	public:
		Storage(T* data, size_t size)
			: m_Data(data), m_Size(size) {
			assert(data != nullptr || size == 0);
		}

		T* Data() {
			return m_Data;
		}

		const T* Data() const {
			return m_Data;
		}

		size_t Size() const {
			return m_Size;
		}

	private:
		T* m_Data;
		size_t m_Size; // Number of elements (NOT size in bytes)
	};

	template <typename T>
	inline Storage<T> MakeStorage(ArenaAllocator& allocator, size_t size) {
		T* ptr = allocator.Allocate<T>(size);
		return Storage<T>(ptr, size);
	}
}