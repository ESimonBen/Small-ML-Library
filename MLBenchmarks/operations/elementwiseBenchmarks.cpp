/// elementwiseOperations.cpp
#include <benchmark/benchmark.h>
#include <mlCore/runtime/context.h>
#include <mlCore/operations/elementwise/elementwise.h>

using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

static void BM_Add_Contiguous(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n} };
	Tensor<float> B{ {n} };
	A.Fill(1.0f);
	B.Fill(2.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto C = Add(A, B);
		benchmark::DoNotOptimize(C.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	/// Reports throughput (bytes/sec) in the output, not just latency —
	/// useful once you're comparing against a SIMD path later.
	state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * sizeof(float) * 3); /// 3 values being processed: A and B being read, C being writter
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_Add_Broadcast(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n, 8} };
	Tensor<float> B{ {1, 8} };

	A.Fill(1.0f);
	B.Fill(2.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto C = Add(A, B);
		benchmark::DoNotOptimize(C.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * sizeof(float) * 3);
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_Subtract_Contiguous(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n} };
	Tensor<float> B{ {n} };
	A.Fill(1.0f);
	B.Fill(2.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto C = Subtract(A, B);
		benchmark::DoNotOptimize(C.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	/// Reports throughput (bytes/sec) in the output, not just latency —
	/// useful once you're comparing against a SIMD path later.
	state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * sizeof(float) * 3); /// 3 values being processed: A and B being read, C being writter
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_Subtract_Broadcast(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n, 8} };
	Tensor<float> B{ {1, 8} };

	A.Fill(1.0f);
	B.Fill(2.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto C = Subtract(A, B);
		benchmark::DoNotOptimize(C.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * sizeof(float) * 3);
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_Multiply_Contiguous(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n} };
	Tensor<float> B{ {n} };
	A.Fill(1.0f);
	B.Fill(2.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto C = Multiply(A, B);
		benchmark::DoNotOptimize(C.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * sizeof(float) * 3);
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_Multiply_Broadcast(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n, 8} };
	Tensor<float> B{ {1, 8} };

	A.Fill(1.0f);
	B.Fill(2.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto C = Multiply(A, B);
		benchmark::DoNotOptimize(C.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * sizeof(float) * 3);
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_Divide_Contiguous(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n} };
	Tensor<float> B{ {n} };
	A.Fill(1.0f);
	B.Fill(2.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto C = Divide(A, B);
		benchmark::DoNotOptimize(C.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * sizeof(float) * 3);
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_Divide_Broadcast(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n, 8} };
	Tensor<float> B{ {1, 8} };

	A.Fill(1.0f);
	B.Fill(2.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto C = Divide(A, B);
		benchmark::DoNotOptimize(C.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * sizeof(float) * 3);
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

/// Sweep from small (latency-dominated, everything fits in L1/L2) to large
/// (bandwidth-dominated, spills out of cache) — the two together tell you
/// where the crossover happens on real hardware.
BENCHMARK(BM_Add_Contiguous)->RangeMultiplier(4)->Range(1 << 6, 1 << 22);
BENCHMARK(BM_Add_Broadcast)->RangeMultiplier(4)->Range(1 << 6, 1 << 18);
BENCHMARK(BM_Subtract_Contiguous)->RangeMultiplier(4)->Range(1 << 6, 1 << 22);
BENCHMARK(BM_Subtract_Broadcast)->RangeMultiplier(4)->Range(1 << 6, 1 << 18);
BENCHMARK(BM_Multiply_Contiguous)->RangeMultiplier(4)->Range(1 << 6, 1 << 22);
BENCHMARK(BM_Multiply_Broadcast)->RangeMultiplier(4)->Range(1 << 6, 1 << 18);
BENCHMARK(BM_Divide_Contiguous)->RangeMultiplier(4)->Range(1 << 6, 1 << 22);
BENCHMARK(BM_Divide_Broadcast)->RangeMultiplier(4)->Range(1 << 6, 1 << 18);

static void RunPower(benchmark::State& state, float exponent) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n} };
	A.Fill(2.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto B = Power(A, 0.0f);
		benchmark::DoNotOptimize(B.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

static void BM_Power_Zero(benchmark::State& state) {
	RunPower(state, 0);
}

static void BM_Power_One(benchmark::State& state) {
	RunPower(state, 1);
}

static void BM_Power_Square(benchmark::State& state) {
	RunPower(state, 2);
}

static void BM_Power_Cube(benchmark::State& state) {
	RunPower(state, 3);
}

BENCHMARK(BM_Power_Zero)->RangeMultiplier(4)->Range(1 << 6, 1 << 18);
BENCHMARK(BM_Power_One)->RangeMultiplier(4)->Range(1 << 6, 1 << 18);
BENCHMARK(BM_Power_Square)->RangeMultiplier(4)->Range(1 << 6, 1 << 18);
BENCHMARK(BM_Power_Cube)->RangeMultiplier(4)->Range(1 << 6, 1 << 18);

static void BM_Abs(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));
	
	Tensor<float> A{ {n} };
	for (size_t i = 0; i < n; ++i) {
		if (i < (n / 2)) {
			A[i] = static_cast<float>(i);
		}
		else {
			A[i] = -(static_cast<float>(i));
		}
	}

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto B = Abs(A);
		benchmark::DoNotOptimize(B.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * sizeof(float) * 2);
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

BENCHMARK(BM_Abs)->RangeMultiplier(4)->Range(1 << 6, 1 << 22);

static void RunClamp(benchmark::State& state, float min, float max) {
	Tensor<float> A{ {1000} };
	
	for (size_t i = 0; i < 1000; ++i) {
		if (i < (1000 / 2)) {
			A[i] = static_cast<float>(i);
		}
		else {
			A[i] = -(static_cast<float>(i));
		}
	}

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto B = Clamp(A, min, max);
		benchmark::DoNotOptimize(B.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * 1000 * sizeof(float) * 2);
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 1000);
}

static void BM_Clamp_WideRange(benchmark::State& state) {
	RunClamp(state, 250, 750);
}

static void BM_Clamp_NarrowRange(benchmark::State& state) {
	RunClamp(state, 375, 625);
}

BENCHMARK(BM_Clamp_WideRange);
BENCHMARK(BM_Clamp_NarrowRange);

static void BM_Log(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n} };
	A.Fill(2.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto B = Log(A);
		benchmark::DoNotOptimize(B.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * sizeof(float) * 2);
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

BENCHMARK(BM_Log)->RangeMultiplier(4)->Range(1 << 6, 1 << 18);

static void BM_Exp(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n} };
	A.Fill(2.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto B = Exp(A);
		benchmark::DoNotOptimize(B.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * 2);
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

BENCHMARK(BM_Exp)->RangeMultiplier(4)->Range(1 << 6, 1 << 18);

static void BM_Equal(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n} };
	Tensor<float> B{ {n} };
	A.Fill(3.0f);
	B.Fill(3.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto C = Equal(A, B);
		benchmark::DoNotOptimize(C.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * sizeof(float) * 3);
	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}

BENCHMARK(BM_Equal)->RangeMultiplier(4)->Range(1 << 6, 1 << 22);