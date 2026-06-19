 /// allocator.inl
#include <new>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <mlCore/config.h>

namespace MLCore::Memory {
	ArenaAllocator::ArenaAllocator(size_t arenaSize)
		: m_ArenaCapacity(arenaSize), m_Offset(0) {
		m_Arena = static_cast<char*>(std::malloc(arenaSize));

		if (!m_Arena) {
			throw std::bad_alloc();
		}

		#ifdef ML_CORE_DEBUG
			std::memset(m_Arena, 0xCD, arenaSize);
		#endif
	}

	ArenaAllocator::~ArenaAllocator() {
		std::free(m_Arena);
		m_Arena = nullptr;
	}

	/// <summary>
	/// Rounds a pointer or offset value forward to the next alignment boundary.
	/// </summary>
	/// <param name="ptr">The pointer value or offset (as size_t) to be aligned.</param>
	/// <param name="alignment">The alignment boundary (in bytes). Must be non-zero and is intended to be a power of two for correct behavior with the bitwise calculation.</param>
	/// <returns>The smallest size_t value greater than or equal to ptr that is aligned to alignment.</returns>
	inline constexpr size_t AlignForward(size_t ptr, size_t alignment) {
		return (ptr + alignment - 1) & ~(alignment - 1);
	}
	
	template <typename T>
	inline T* ArenaAllocator::Allocate(size_t size) {
		assert(m_Arena != nullptr); ///  Make sure the arena is initialized

		if (size == 0) {
			return nullptr;
		}

		size_t alignment = alignof(T);
		assert(alignment > 0);

		uintptr_t currentAddress = reinterpret_cast<uintptr_t>(m_Arena) + m_Offset;
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
			std::memset(result, 0xCD, requiredBytes); /// Set each byte of "result" to the value 0xCD for "requiredBytes" number of bytes
		#endif

		m_Offset += requiredBytes;

		return result;
	}

	void ArenaAllocator::Reset() {
		#ifdef ML_CORE_DEBUG
			std::memset(m_Arena, 0xDD, m_ArenaCapacity);
		#endif

		m_Offset = 0;
	}

	size_t ArenaAllocator::Capacity() const {
		return m_ArenaCapacity;
	}

	size_t ArenaAllocator::UsedBytes() const {
		return m_Offset;
	}
	
	size_t ArenaAllocator::Remaining() const {
		return m_ArenaCapacity - m_Offset;
	}
	
	bool ArenaAllocator::IsInitialized() const {
		return m_Arena != nullptr;
	}
}