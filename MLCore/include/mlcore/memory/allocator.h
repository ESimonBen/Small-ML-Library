// allocator.h
#pragma once

namespace mlcore::memory {
	class ArenaAllocator {
	public:
		ArenaAllocator(size_t arenaSize = 1024 * 1024); // Allocate a default size of 1 MB
		~ArenaAllocator();
		ArenaAllocator(const ArenaAllocator&) = delete;
		ArenaAllocator& operator=(const ArenaAllocator&) = delete;

		// Allocate raw memory of N elements of type T
		template <typename T>
		T* allocate(size_t size);

		// Reset the arena (Arenas often don't have individual deallocation)
		void reset();

		// Get the capacity
		size_t capacity() const;

		// Get amount of memory used
		size_t used() const;

		// Get the amount of memory remaining
		size_t remaining() const;

	private:
		char* m_Arena;
		size_t m_ArenaCapacity;
		size_t m_Offset;
	};
}

#include "mlcore/memory/allocator.inl"