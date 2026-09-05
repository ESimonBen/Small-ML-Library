/// reductionBenchmarks.cpp
#include <benchmark/benchmark.h>
#include <mlCore/runtime/context.h>
#include <mlCore/operations/reduction/reduction.h>

using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

static void BM_SumAll(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n} };
	A.Fill(0.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto B = SumAll(A);
		benchmark::DoNotOptimize(B.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));

	MLCore::Runtime::MLContext::GetAllocator().Reset();
}

static void BM_MeanAll(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n} };
	A.Fill(0.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto B = MeanAll(A);
		benchmark::DoNotOptimize(B.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));

	MLCore::Runtime::MLContext::GetAllocator().Reset();
}

static void BM_MaxAll(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n} };
	size_t size = A.NumElements();

	for (size_t i = 0; i < size; ++i) {
		A[i] = static_cast<float>(i);
	}

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto B = MaxAll(A);
		benchmark::DoNotOptimize(B.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));

	MLCore::Runtime::MLContext::GetAllocator().Reset();
}

static void BM_MinAll(benchmark::State& state) {
	const size_t n = static_cast<size_t>(state.range(0));

	Tensor<float> A{ {n} };
	size_t size = A.NumElements();

	for (size_t i = 0; i < size; ++i) {
		A[i] = static_cast<float>(i);
	}

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto B = MinAll(A);
		benchmark::DoNotOptimize(B.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));

	MLCore::Runtime::MLContext::GetAllocator().Reset();
}

BENCHMARK(BM_SumAll)->RangeMultiplier(4)->Range(1 << 8, 1 << 22);
BENCHMARK(BM_MeanAll)->RangeMultiplier(4)->Range(1 << 8, 1 << 22);
BENCHMARK(BM_MaxAll)->RangeMultiplier(4)->Range(1 << 8, 1 << 21);
BENCHMARK(BM_MinAll)->RangeMultiplier(4)->Range(1 << 8, 1 << 21);

static void RunAxisSum(benchmark::State& state, size_t dim0, size_t dim1, size_t axis) {
	Tensor<float> A{ {dim0, dim1} };
	A.Fill(1.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto B = AxisSum(A, axis, true);
		benchmark::DoNotOptimize(B.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(dim0 * dim1));

	MLCore::Runtime::MLContext::GetAllocator().Reset();
}

static void BM_AxisSum_OuterAxis(benchmark::State& state) {
	RunAxisSum(state, 1024, 1024, 0);
}

static void BM_AxisSum_InnerAxis(benchmark::State& state) {
	RunAxisSum(state, 1024, 1024, 1);
}

BENCHMARK(BM_AxisSum_OuterAxis);
BENCHMARK(BM_AxisSum_InnerAxis);

static void RunAxisMean(benchmark::State& state, size_t dim0, size_t dim1, size_t axis) {
	Tensor<float> A{ {dim0, dim1} };
	A.Fill(1.0f);

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto B = AxisMean(A, axis, true);
		benchmark::DoNotOptimize(B.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(dim0 * dim1));

	MLCore::Runtime::MLContext::GetAllocator().Reset();
}

static void BM_AxisMean_OuterAxis(benchmark::State& state) {
	RunAxisMean(state, 1024, 1024, 0);
}

static void BM_AxisMean_InnerAxis(benchmark::State& state) {
	RunAxisMean(state, 1024, 1024, 1);
}

BENCHMARK(BM_AxisMean_OuterAxis);
BENCHMARK(BM_AxisMean_InnerAxis);

static void RunAxisMax(benchmark::State& state, size_t dim0, size_t dim1, size_t axis) {
	Tensor<float> A{ {dim0, dim1} };
	size_t size = A.NumElements();

	for (size_t i = 0; i < size; ++i) {
		A[i] = static_cast<float>(i);
	}

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto B = AxisMax(A, axis, true);
		benchmark::DoNotOptimize(B.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(dim0 * dim1));

	MLCore::Runtime::MLContext::GetAllocator().Reset();
}

static void BM_AxisMax_OuterAxis(benchmark::State& state) {
	RunAxisMax(state, 1024, 1024, 0);
}

static void BM_AxisMax_InnerAxis(benchmark::State& state) {
	RunAxisMax(state, 1024, 1024, 1);
}

BENCHMARK(BM_AxisMax_OuterAxis);
BENCHMARK(BM_AxisMax_InnerAxis);

static void RunAxisMin(benchmark::State& state, size_t dim0, size_t dim1, size_t axis) {
	Tensor<float> A{ {dim0, dim1} };
	size_t size = A.NumElements();

	for (size_t i = 0; i < size; ++i) {
		A[i] = static_cast<float>(i);
	}

	size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

	for (auto _ : state) {
		auto B = AxisMin(A, axis, true);
		benchmark::DoNotOptimize(B.Data());

		state.PauseTiming();
		MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
		state.ResumeTiming();
	}

	state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(dim0 * dim1));

	MLCore::Runtime::MLContext::GetAllocator().Reset();
}

static void BM_AxisMin_OuterAxis(benchmark::State& state) {
	RunAxisMin(state, 1024, 1024, 0);
}

static void BM_AxisMin_InnerAxis(benchmark::State& state) {
	RunAxisMin(state, 1024, 1024, 1);
}

BENCHMARK(BM_AxisMin_OuterAxis);
BENCHMARK(BM_AxisMin_InnerAxis);