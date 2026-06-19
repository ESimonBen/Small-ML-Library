 /// storage.h
#pragma once
#include <cstddef>
#include <mlCore/memory/allocator.h>

namespace MLCore::Memory {
	/// <summary>
	/// A lightweight, non-owning storage wrapper that holds a pointer to an array of T and the number of elements. Provides shallow copy and move semantics and accessors for the pointer and size.
	/// </summary>
	/// <typeparam name="T">Type of the elements referenced by the storage.</typeparam>
	template <typename T>
	class Storage {
	public:
		/// <summary>
		/// Initializes a Storage object with a pointer to data and its size. Asserts that data is non-null when size is greater than zero.
		/// </summary>
		/// <param name="data">Pointer to the first element of the storage. May be nullptr only if size is 0.</param>
		/// <param name="size">Number of elements referenced by data.</param>
		Storage(T* data, size_t size)
			: m_Data(data), m_Size(size) {
			assert(data != nullptr || size == 0);
		}

		/// <summary>
		/// Copy constructor that initializes a new Storage object by copying the data pointer and size from another Storage instance. This operation is noexcept.
		/// </summary>
		/// <param name="other">The Storage instance to copy from; its m_Data and m_Size are copied into the new object.</param>
		Storage(const Storage& other) noexcept
			: m_Data(other.m_Data), m_Size(other.m_Size)
		{}

		/// <summary>
		/// Copy assignment operator that assigns the contents of another Storage to this instance, with a self-assignment check.
		/// </summary>
		/// <param name="other">The source Storage to copy from. If other is the same object, no assignment is performed.</param>
		/// <returns>Reference to this Storage instance (*this) after assignment.</returns>
		Storage& operator=(const Storage& other) noexcept {
			if (this != &other) {
				m_Data = other.m_Data;
				m_Size = other.m_Size;
			}

			return *this;
		}

		/// <summary>
		/// Move constructor that transfers ownership of the internal data and size from another Storage instance to the newly constructed object. The operation is noexcept and leaves the source in a valid empty state.
		/// </summary>
		/// <param name="other">Rvalue reference to the Storage instance to move from. Its internal pointer and size are transferred to the new object; after the move, other.m_Data is set to nullptr and other.m_Size is set to 0.</param>
		Storage(Storage&& other) noexcept
			: m_Data(other.m_Data), m_Size(other.m_Size) {
			other.m_Data = nullptr;
			other.m_Size = 0;
		}

		/// <summary>
		/// Move assignment operator that transfers ownership of internal resources from other to this Storage instance. It checks for self-assignment and is noexcept.
		/// </summary>
		/// <param name="other">Rvalue reference to a Storage object to move from. The operator moves internal data and size fields to this object and leaves other in an empty state (m_Data == nullptr, m_Size == 0).</param>
		/// <returns>Reference to this Storage object after the move (i.e., *this).</returns>
		Storage& operator=(Storage&& other) noexcept {
			if (this != &other) {
				m_Data = other.m_Data;
				other.m_Data = nullptr;
				m_Size = other.m_Size;
				other.m_Size = 0;
			}

			return *this;
		}

		/// <summary>
		/// Returns a pointer to the object's internal data buffer.
		/// </summary>
		/// <returns>A pointer to the stored data (m_Data).</returns>
		T* Data() {
			return m_Data;
		}

		/// <summary>
		/// Returns a read-only pointer to the object's internal data buffer.
		/// </summary>
		/// <returns>A pointer to the internal data (const T*). The returned pointer provides read-only access to m_Data, is not owned by the caller, and remains valid only while the object exists and is not modified in a way that invalidates its storage.</returns>
		const T* Data() const {
			return m_Data;
		}

		/// <summary>
		/// Returns the current size stored in the object without modifying it.
		/// </summary>
		/// <returns>The stored size as a size_t value (the value of m_Size).</returns>
		size_t Size() const {
			return m_Size;
		}

	private:
		T* m_Data; /// A pointer to an object or array of type T.
		size_t m_Size; /// Member variable that holds a size or count.
	};

	/// <summary>
	/// Allocates memory for a specified number of elements using the provided arena allocator and returns a Storage<T> representing the allocated buffer.
	/// </summary>
	/// <typeparam name="T">The element type stored in the allocated buffer.</typeparam>
	/// <param name="allocator">The ArenaAllocator used to allocate memory for the elements.</param>
	/// <param name="size">The number of elements of type T to allocate.</param>
	/// <returns>A Storage<T> constructed from the allocated raw pointer and the element count (the buffer pointer and its size).</returns>
	template <typename T>
	inline Storage<T> MakeStorage(ArenaAllocator& allocator, size_t size) {
		T* raw = allocator.Allocate<T>(size);
		return Storage<T>(raw, size);
	}
}