// allocator.cpp
#include <mlcore/memory/allocator.h>

namespace mlcore::memory {
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

	void ArenaAllocator::reset() {
		#ifdef ML_CORE_DEBUG
			std::memset(m_Arena, 0xDD, m_ArenaCapacity);
		#endif

		m_Offset = 0;
	}

	size_t ArenaAllocator::capacity() const {
		return m_ArenaCapacity;
	}

	size_t ArenaAllocator::used() const {
		return m_Offset;
	}

	size_t ArenaAllocator::remaining() const {
		return m_ArenaCapacity - m_Offset;
	}
}