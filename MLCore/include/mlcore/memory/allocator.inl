// allocator.inl
#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <mlcore/config.h>

namespace MLCore::Memory {
	inline constexpr size_t AlignForward(size_t ptr, size_t alignment) {
		return (ptr + alignment - 1) & ~(alignment - 1);
	}

	template <typename T>
	inline T* ArenaAllocator::Allocate(size_t size) {
		assert(m_Arena != nullptr); //  Make sure the arena is initialized

		if (size == 0) {
			return nullptr;
		}

		size_t alignment = alignof(T);
		assert(alignment > 0);

		uintptr_t currentAddress = reinterpret_cast<size_t>(m_Arena) + m_Offset;
		uintptr_t alignedAddress = AlignForward(currentAddress, alignment);

		size_t adjustment = alignedAddress - currentAddress;
		size_t requiredBytes = sizeof(T) * size;

		// Bounds check
		if (m_Offset + adjustment + requiredBytes > m_ArenaCapacity) {
			throw std::bad_alloc();
		}

		m_Offset += adjustment;

		T* result = reinterpret_cast<T*>(m_Arena + m_Offset);

		#ifdef ML_CORE_DEBUG
			std::memset(result, 0xCD, requiredBytes); // Set each byte of "result" to the value 0xCD for "requiredBytes" number of bytes
		#endif

		m_Offset += requiredBytes;

		return result;
	}
}