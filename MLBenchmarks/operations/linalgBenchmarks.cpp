/// linalgBenchmarks.cpp
#include <benchmark/benchmark.h>
#include <mlCore/runtime/context.h>
#include <mlCore/operations/linearAlgebra/linalg.h>

using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

/// Basic function to run matrix multiplication operation. Size is passed in from benchmark tests.
static void RunMatMul(benchmark::State& state, size_t M, size_t K, size_t N) {
	Tensor<float> A{ {M, K} };
	Tensor<float> B{ {K, N} };
	A.Fill(1.0f);
	B.Fill(2.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto C = MatMultiply(A, B);
		benchmark::DoNotOptimize(C.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(2 * M * K * N));
}

/// Square Benchmarks
static void BM_MatMul_Square_64(benchmark::State& state) {
	RunMatMul(state, 64, 64, 64);
}

static void BM_MatMul_Square_256(benchmark::State& state) {
	RunMatMul(state, 256, 256, 256);
}

static void BM_MatMul_Square_1024(benchmark::State& state) {
	RunMatMul(state, 1024, 1024, 1024);
}

BENCHMARK(BM_MatMul_Square_64);
BENCHMARK(BM_MatMul_Square_256);
BENCHMARK(BM_MatMul_Square_1024);

/// Linear Layer Benchmarks
static void BM_MatMul_LinearLayer_Small(benchmark::State& state) {
	RunMatMul(state, 32, 128, 256);
}

static void BM_MatMul_LinearLayer_Medium(benchmark::State& state) {
	RunMatMul(state, 128, 512, 512);
}

BENCHMARK(BM_MatMul_LinearLayer_Small);
BENCHMARK(BM_MatMul_LinearLayer_Medium);

static void BM_MatMul_TallSkinny(benchmark::State& state) {
	RunMatMul(state, 4096, 64, 8);
}

static void BM_MatMul_ShortFat(benchmark::State& state) {
	RunMatMul(state, 8, 64, 4096);
}

BENCHMARK(BM_MatMul_TallSkinny);
BENCHMARK(BM_MatMul_ShortFat);