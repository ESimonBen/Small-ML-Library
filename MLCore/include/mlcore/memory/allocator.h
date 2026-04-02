// allocator.h
#pragma once

namespace MLCore::Memory {
	class ArenaAllocator {
	public:
		ArenaAllocator(size_t arenaSize = 1024 * 1024); // Allocate a default size of 1 MB
		~ArenaAllocator();
		ArenaAllocator(const ArenaAllocator&) = delete;
		ArenaAllocator& operator=(const ArenaAllocator&) = delete;

		// Allocate raw memory of N elements of type T
		template <typename T>
		T* Allocate(size_t size);

		// Reset the arena (Arenas often don't have individual deallocation)
		void Reset();

		// Get the Capacity
		size_t Capacity() const;

		// Get amount of memory used
		size_t UsedBytes() const;

		// Get the amount of memory Remaining
		size_t Remaining() const;

	private:
		char* m_Arena;
		size_t m_ArenaCapacity;
		size_t m_Offset;
	};
}

#include <mlCore/memory/allocator.inl>