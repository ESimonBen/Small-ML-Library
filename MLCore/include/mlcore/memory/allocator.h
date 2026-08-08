 /// allocator.h
#pragma once

namespace MLCore::Memory {
	/// <summary>
	/// A simple arena (region) allocator that provides fast linear allocation from a contiguous block of memory. The allocator reserves a fixed-capacity buffer and hands out raw memory without individual deallocations; allocations are reclaimed by calling Reset.
	/// </summary>
	class ArenaAllocator {
	public:
		/// <summary>
		/// Constructs an ArenaAllocator by allocating a contiguous memory arena of the specified size. Initializes internal capacity and offset, and (when ML_CORE_DEBUG is defined) fills the arena with 0xCD. Throws std::bad_alloc if allocation fails.
		/// </summary>
		/// <param name="arenaSize">The size in bytes of the arena to allocate.</param>
		ArenaAllocator(size_t arenaSize = 1024 * 1024 * 1024);

		/// <summary>
		/// Destructor for ArenaAllocator that frees the arena memory and clears the internal pointer.
		/// </summary>
		~ArenaAllocator();

		ArenaAllocator(const ArenaAllocator&) = delete;
		ArenaAllocator& operator=(const ArenaAllocator&) = delete;

		/// <summary>
		/// Allocates raw, properly aligned storage for up to 'size' elements of type T from this ArenaAllocator. Does not construct objects. Returns a pointer to the allocated memory or nullptr if size is zero. Asserts that the arena is initialized and throws std::bad_alloc if there is not enough capacity.
		/// </summary>
		/// <typeparam name="T">Element type to allocate. Its size and alignment (alignof(T)) are used for allocation. Note that constructors are not invoked for the returned storage.</typeparam>
		/// <param name="size">Number of elements of type T to allocate. If size is 0, the function returns nullptr. The function performs bounds checks and will throw std::bad_alloc if the arena lacks sufficient space.</param>
		/// <returns>A pointer to the first element of the allocated, properly aligned raw storage for T. Returns nullptr when size is 0. On allocation failure due to insufficient arena capacity, std::bad_alloc is thrown. In debug builds the returned memory is filled with 0xCD; otherwise it is uninitialized.</returns>
		template <typename T>
		T* Allocate(size_t size);

		/// <summary>
		/// Resets the allocator to its initial state. In debug builds, fills the arena memory with 0xDD and sets the internal offset back to 0. This does not free the underlying memory buffer.
		/// </summary>
		void Reset();

		/// <summary>
		/// Returns the total capacity of the arena allocator.
		/// </summary>
		/// <returns>The arena's capacity as a size_t value.</returns>
		size_t Capacity() const;

		/// <summary>
		/// Returns the number of bytes currently used by the arena allocator.
		/// </summary>
		/// <returns>The number of bytes used in the arena (size_t), as represented by m_Offset.</returns>
		size_t UsedBytes() const;

		/// <summary>
		/// Returns the number of bytes remaining in the arena allocator without modifying its state.
		/// </summary>
		/// <returns>The number of bytes available for allocation (computed as m_ArenaCapacity - m_Offset).</returns>
		size_t Remaining() const;

		/// <summary>
		/// Checks whether the allocator has been initialized (i.e., has a non-null arena).
		/// </summary>
		/// <returns>true if the allocator is initialized (m_Arena is not null); otherwise false.</returns>
		bool IsInitialized() const;

	private:
		char* m_Arena; /// Pointer to a memory arena buffer.
		size_t m_ArenaCapacity; /// Holds the total capacity of the memory arena, in bytes.
		size_t m_Offset; /// A size_t variable that stores an offset value.
	};
}

#include "allocator.inl"