// allocator.inl
#include <new>
#include <cstddef>
#include <cstdlib>
#include <mlcore/config.h>

namespace mlcore::memory {
	inline constexpr size_t alignForward(size_t ptr, size_t alignment) {
		return (ptr + alignment - 1) & ~(alignment - 1);
	}

	template <typename T>
	inline T* ArenaAllocator::allocate(size_t size) {
		if (size == 0) {
			return nullptr;
		}

		size_t alignment = alignof(T);

		uintptr_t currentAddress = reinterpret_cast<size_t>(m_Arena) + m_Offset;
		uintptr_t alignedAddress = alignForward(currentAddress, alignment);

		size_t adjustment = alignedAddress - currentAddress;
		size_t requiredBytes = sizeof(T) * size;

		// Bounds check
		if (offset + adjustment + requiredBytes > m_ArenaCapacity) {
			throw std::bad_alloc();
		}

		offset += adjustment;

		T* result = reinterpret_cast<T*>(m_Arena + m_Offset);

		#ifdef ML_CORE_DEBUG
			std::memset(result, 0xCD, requiredBytes); // Set each byte of "result" to the value 0xCD for "requiredBytes" number of bytes
		#endif

		offset += requiredBytes;

		return result;
	}
}