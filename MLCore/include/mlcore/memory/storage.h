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

		Storage(const Storage& other) noexcept
			: m_Data(other.m_Data), m_Size(other.m_Size)
		{}

		Storage& operator=(const Storage& other) noexcept {
			if (this != &other) {
				m_Data = other.m_Data;
				m_Size = other.m_Size;
			}

			return *this;
		}

		Storage(Storage&& other) noexcept
			: m_Data(other.m_Data), m_Size(other.m_Size) {
			other.m_Data = nullptr;
			other.m_Size = 0;
		}

		Storage& operator=(Storage&& other) noexcept {
			if (this != &other) {
				m_Data = other.m_Data;
				other.m_Data = nullptr;
				m_Size = other.m_Size;
				other.m_Size = 0;
			}

			return *this;
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
		T* raw = allocator.Allocate<T>(size);
		return Storage<T>(raw, size);
	}
}