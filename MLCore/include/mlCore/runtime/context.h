/// context.h
#pragma once
#include <random>
#include <atomic>
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

		/// <summary>
		/// Returns a reference to a shared (singleton) Memory::ArenaAllocator instance.
		/// </summary>
		/// <returns>A reference to a function-local static Memory::ArenaAllocator that is constructed on first call and remains alive for the program's lifetime.</returns>
		static Memory::ArenaAllocator& GetSharedAllocator() {
			static Memory::ArenaAllocator sharedAllocator;
			return sharedAllocator;
		}

		/// <summary>
		/// Returns this thread's random engine, lazily constructed on first use per thread
		/// and persisting across calls (unlike constructing a new engine per Init() call).
		/// </summary>
		static std::mt19937& GetRNG() {
			thread_local std::mt19937 gen(MakeThreadSeed());
			return gen;
		}

		/// <summary>
		/// Sets a deterministic base seed used by any thread-local RNG constructed AFTER this call.
		/// Must be called before the first GetRNG() on a given thread to take effect there,
		/// since thread_local construction happens lazily on first access.
		/// </summary>
		static void SetGlobalSeed(uint32_t seed) {
			s_GlobalSeed = seed;
			s_HasGlobalSeed = true;
		}

		/// <summary>
		/// Immediately reseeds the CALLING thread's RNG. Use this to reset an
		/// already-constructed thread's stream (e.g. between independent experiment runs)
		/// without needing a global seed set in advance.
		/// </summary>
		static void SeedCurrentThread(uint32_t seed) {
			GetRNG().seed(seed);
		}

	private:
		/// Private default constructor
		MLContext() = default;

		/// <summary>
		/// Generates a 32-bit seed suitable for per-thread random number generation.
		/// </summary>
		/// <returns>A 32-bit seed value. If s_HasGlobalSeed is true, returns s_GlobalSeed plus a distinct per-call increment from an internal atomic counter (deterministic and distinct per thread, and increments the counter). Otherwise, returns a value obtained from std::random_device.</returns>
		static uint32_t MakeThreadSeed() {
			if (s_HasGlobalSeed) {
				/// Distinct-but-deterministic per-thread seed, so multiple threads
				/// don't produce identical streams under one global seed.
				static std::atomic<uint32_t> counter{ 0 };
				return s_GlobalSeed + counter.fetch_add(1, std::memory_order_relaxed);
			}

			std::random_device rd;
			return rd();
		}

		static inline std::atomic<bool> s_HasGlobalSeed{ false }; /// Thread-safe flag indicating whether a global seed has been set.
		static inline std::atomic<uint32_t> s_GlobalSeed{ 0 }; /// A global atomic 32-bit unsigned integer seed initialized to 0.
	};
}