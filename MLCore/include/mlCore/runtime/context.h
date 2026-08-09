/// context.h
#pragma once
#include <mlCore/memory/allocator.h>

namespace MLCore::Runtime {
	/// <summary>
	/// Singleton context that owns and provides access to a CPU memory arena allocator.
	/// </summary>
	class MLContext {
	public:
		/// Deleted copy constructor
		MLContext(const MLContext&) = delete;

		/// Deleted move constructor
		MLContext(MLContext&&) = delete;

		/// <summary>
		/// Returns a reference to a shared MLContext instance, created on first use (function-local static).
		/// </summary>
		/// <returns>A reference to the single MLContext object. The instance is initialized on first call and has static lifetime; initialization is thread-safe in C++11 and later.</returns>
		static MLContext& GetContext() {
			static MLContext context;
			return context;
		}

		/// <summary>
		/// Returns a reference to a thread-local Memory::ArenaAllocator instance.
		/// </summary>
		/// <returns>A reference to a thread-local Memory::ArenaAllocator. The allocator is constructed on first call per thread and persists until the thread exits.</returns>
		static Memory::ArenaAllocator& GetAllocator() {
			thread_local Memory::ArenaAllocator threadAllocator;
			return threadAllocator;
		}

	private:
		/// Private default constructor
		MLContext() = default;
	};
}