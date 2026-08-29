 /// allocator.inl
#include <new>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <cstring>

namespace MLCore::Memory {
	inline ArenaAllocator::ArenaAllocator(size_t arenaSize)
		: m_ArenaCapacity(arenaSize), m_Offset(0) {
		m_Arena = static_cast<char*>(std::malloc(arenaSize));

		if (!m_Arena) {
			throw std::bad_alloc();
		}

		#ifdef MLCORE_DEBUG
			std::memset(m_Arena, 0xCD, arenaSize);
		#endif
	}

	inline ArenaAllocator::~ArenaAllocator() {
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
		///  Make sure the arena is initialized
		if (!m_Arena) {
			throw std::bad_alloc();
		}

		if (size == 0) {
			return nullptr;
		}

		size_t alignment = alignof(T);

		uintptr_t currentAddress = reinterpret_cast<uintptr_t>(m_Arena) + m_Offset;
		uintptr_t alignedAddress = AlignForward(currentAddress, alignment);

		size_t adjustment = alignedAddress - currentAddress;
		size_t requiredBytes = sizeof(T) * size;

		/// Bounds check
		if (m_Offset + adjustment + requiredBytes > m_ArenaCapacity) {
			throw std::bad_alloc();
		}

		m_Offset += adjustment;

		T* result = reinterpret_cast<T*>(m_Arena + m_Offset);

		#ifdef MLCORE_DEBUG
			std::memset(result, 0xCD, requiredBytes); /// Set each byte of "result" to the value 0xCD for "requiredBytes" number of bytes
		#endif

		m_Offset += requiredBytes;

		return result;
	}

	inline void ArenaAllocator::Reset() {
		#ifdef MLCORE_DEBUG
			assert(m_DebugPersistentRanges.empty() && "Reset() would reclaim allocations registered as persistent — persistent objects (Parameter, TensorDataset) must live on the shared arena, never a thread-local scratch arena.");
			std::memset(m_Arena, 0xDD, m_ArenaCapacity);
		#endif

		m_Offset = 0;
	}

	inline size_t ArenaAllocator::Capacity() const {
		return m_ArenaCapacity;
	}

	inline size_t ArenaAllocator::UsedBytes() const {
		return m_Offset;
	}
	
	inline size_t ArenaAllocator::Remaining() const {
		return m_ArenaCapacity - m_Offset;
	}
	
	inline bool ArenaAllocator::IsInitialized() const {
		return m_Arena != nullptr;
	}
	
	inline size_t ArenaAllocator::Checkpoint() const {
		return m_Offset;
	}
	
	inline void ArenaAllocator::RestoreCheckpoint(size_t checkpoint) {
		if (checkpoint > m_Offset) {
			throw std::out_of_range("ERROR: RestoreCheckpoint: Cannot restore checkpoint in unitialized memory");
		}

		#ifdef MLCORE_DEBUG
			const uintptr_t base = reinterpret_cast<uintptr_t>(m_Arena);

			for (const auto& [ptr, bytes]: m_DebugPersistentRanges) {
				const uintptr_t rangeOffset = reinterpret_cast<uintptr_t>(ptr) - base;
				assert(rangeOffset + bytes <= checkpoint && "RestoreCheckpoint would reclaim a registered persistent allocation — the checkpoint was taken before this object was constructed");
			}
			
			std::memset(m_Arena + checkpoint, 0xDD, m_Offset - checkpoint);
		#endif

		m_Offset = checkpoint;
	}

	#ifdef MLCORE_DEBUG
	inline void ArenaAllocator::RegisterPersistent(const void* ptr, size_t bytes) {
		m_DebugPersistentRanges.emplace_back(ptr, bytes);
	}
	#endif
}