/// convolutionBenchmarks.cpp
#include <benchmark/benchmark.h>
#include <mlCore/runtime/context.h>
#include <mlCore/operations/convolution/convolution.h>

using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

static void RunConv1D(benchmark::State& state, size_t batch, size_t inChannels, size_t inL, size_t outChannels, size_t kL,
                      size_t strideL = 1, size_t padL = 0) {
    Tensor<float> input{ {batch, inChannels, inL} };
    Tensor<float> kernel{ {outChannels, inChannels, kL} };
    Tensor<float> bias{ {outChannels} };
    input.Fill(1.0f);
    kernel.Fill(0.01f);
    bias.Fill(0.0f);

    size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

    size_t outL = (inL + 2 * padL - kL) / strideL + 1;

    for (auto _ : state) {
        auto output = Conv1D(input, kernel, &bias, strideL, padL);
        benchmark::DoNotOptimize(output.Data());

        state.PauseTiming();
        MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
        state.ResumeTiming();
    }

    int64_t flopsPerOutput = static_cast<int64_t>(inChannels * kL) * 2;
    int64_t numOutputs = static_cast<int64_t>(batch * outChannels * outL);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * numOutputs * flopsPerOutput);
}

static void BM_Conv1D_EarlyLayer(benchmark::State& state) {
    RunConv1D(state, 1, 3, 32, 16, 3);
}

static void BM_Conv1D_DeepLayer(benchmark::State& state) {
    RunConv1D(state, 1, 64, 16, 64, 3);
}

BENCHMARK(BM_Conv1D_EarlyLayer);
BENCHMARK(BM_Conv1D_DeepLayer);

static void BM_Conv1D_Channels_8(benchmark::State& state) {
    RunConv1D(state, 1, 8, 16, 8, 3);
}

static void BM_Conv1D_Channels_32(benchmark::State& state) {
    RunConv1D(state, 1, 32, 16, 32, 3);
}

static void BM_Conv1D_Channels_128(benchmark::State& state) {
    RunConv1D(state, 1, 128, 16, 128, 3);
}

BENCHMARK(BM_Conv1D_Channels_8);
BENCHMARK(BM_Conv1D_Channels_32);
BENCHMARK(BM_Conv1D_Channels_128);

static void BM_Conv1D_Spatial_16(benchmark::State& state) {
    RunConv1D(state, 1, 16, 16, 16, 3);
}

static void BM_Conv1D_Spatial_64(benchmark::State& state) {
    RunConv1D(state, 1, 16, 64, 16, 3);
}

static void BM_Conv1D_Spatial_256(benchmark::State& state) {
    RunConv1D(state, 1, 16, 256, 16, 3);
}

BENCHMARK(BM_Conv1D_Spatial_16);
BENCHMARK(BM_Conv1D_Spatial_64);
BENCHMARK(BM_Conv1D_Spatial_256);

static void BM_Conv1D_Kernel_1x1(benchmark::State& state) {
    RunConv1D(state, 1, 32, 32, 32, 1);
}

static void BM_Conv1D_Kernel_3x3(benchmark::State& state) {
    RunConv1D(state, 1, 32, 32, 32, 3);
}

static void BM_Conv1D_Kernel_5x5(benchmark::State& state) {
    RunConv1D(state, 1, 32, 32, 32, 5);
}

BENCHMARK(BM_Conv1D_Kernel_1x1);
BENCHMARK(BM_Conv1D_Kernel_3x3);
BENCHMARK(BM_Conv1D_Kernel_5x5);

static void BM_Conv1D_Batch_1(benchmark::State& state) {
    RunConv1D(state, 1, 16, 32, 16, 3);
}

static void BM_Conv1D_Batch_8(benchmark::State& state) {
    RunConv1D(state, 8, 16, 32, 16, 3);
}

static void BM_Conv1D_Batch_32(benchmark::State& state) {
    RunConv1D(state, 32, 16, 32, 16, 3);
}

BENCHMARK(BM_Conv1D_Batch_1);
BENCHMARK(BM_Conv1D_Batch_8);
BENCHMARK(BM_Conv1D_Batch_32);

static void BM_Conv1D_StridePad(benchmark::State& state) {
    RunConv1D(state, 1, 16, 64, 16, 3, 2, 1);
}

BENCHMARK(BM_Conv1D_StridePad);

static void RunConv2D(benchmark::State& state, size_t batch, size_t inChannels, size_t inH, size_t inW, size_t outChannels, size_t kH, size_t kW,
                      size_t strideH = 1, size_t strideW = 1, size_t padH = 0, size_t padW = 0) {
    Tensor<float> input{ {batch, inChannels, inH, inW} };
    Tensor<float> kernel{ {outChannels, inChannels, kH, kW} };
    Tensor<float> bias{ {outChannels} };
    input.Fill(1.0f);
    kernel.Fill(0.01f);
    bias.Fill(0.0f);

    size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

    size_t outH = (inH + 2 * padH - kH) / strideH + 1;
    size_t outW = (inW + 2 * padW - kW) / strideW + 1;

    for (auto _ : state) {
        auto output = Conv2D(input, kernel, &bias, strideH, strideW, padH, padW);
        benchmark::DoNotOptimize(output.Data());

        state.PauseTiming();
        MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
        state.ResumeTiming();
    }

    int64_t flopsPerOutput = static_cast<int64_t>(inChannels * kH * kW) * 2;
    int64_t numOutputs = static_cast<int64_t>(batch * outChannels * outH * outW);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * numOutputs * flopsPerOutput);
}

static void BM_Conv2D_EarlyLayer(benchmark::State& state) {
    RunConv2D(state, 1, 3, 32, 32, 16, 3, 3);
}

static void BM_Conv2D_DeepLayer(benchmark::State& state) {
    RunConv2D(state, 1, 64, 16, 16, 64, 3, 3);
}

BENCHMARK(BM_Conv2D_EarlyLayer);
BENCHMARK(BM_Conv2D_DeepLayer);

static void BM_Conv2D_Channels_8(benchmark::State& state) {
    RunConv2D(state, 1, 8, 16, 16, 8, 3, 3);
}

static void BM_Conv2D_Channels_32(benchmark::State& state) {
    RunConv2D(state, 1, 32, 16, 16, 32, 3, 3);
}

static void BM_Conv2D_Channels_128(benchmark::State& state) {
    RunConv2D(state, 1, 128, 16, 16, 128, 3, 3);
}

BENCHMARK(BM_Conv2D_Channels_8);
BENCHMARK(BM_Conv2D_Channels_32);
BENCHMARK(BM_Conv2D_Channels_128);

static void BM_Conv2D_Spatial_16(benchmark::State& state) {
    RunConv2D(state, 1, 16, 16, 16, 16, 3, 3);
}

static void BM_Conv2D_Spatial_64(benchmark::State& state) {
    RunConv2D(state, 1, 16, 64, 64, 16, 3, 3);
}

static void BM_Conv2D_Spatial_256(benchmark::State& state) {
    RunConv2D(state, 1, 16, 256, 256, 16, 3, 3);
}

BENCHMARK(BM_Conv2D_Spatial_16);
BENCHMARK(BM_Conv2D_Spatial_64);
BENCHMARK(BM_Conv2D_Spatial_256);

static void BM_Conv2D_Kernel_1x1(benchmark::State& state) {
    RunConv2D(state, 1, 32, 32, 32, 32, 1, 1);
}

static void BM_Conv2D_Kernel_3x3(benchmark::State& state) {
    RunConv2D(state, 1, 32, 32, 32, 32, 3, 3);
}

static void BM_Conv2D_Kernel_5x5(benchmark::State& state) {
    RunConv2D(state, 1, 32, 32, 32, 32, 5, 5);
}

BENCHMARK(BM_Conv2D_Kernel_1x1);
BENCHMARK(BM_Conv2D_Kernel_3x3);
BENCHMARK(BM_Conv2D_Kernel_5x5);

static void BM_Conv2D_Batch_1(benchmark::State& state) {
    RunConv2D(state, 1, 16, 32, 32, 16, 3, 3);
}

static void BM_Conv2D_Batch_8(benchmark::State& state) {
    RunConv2D(state, 8, 16, 32, 32, 16, 3, 3);
}

static void BM_Conv2D_Batch_32(benchmark::State& state) {
    RunConv2D(state, 32, 16, 32, 32, 16, 3, 3);
}

BENCHMARK(BM_Conv2D_Batch_1);
BENCHMARK(BM_Conv2D_Batch_8);
BENCHMARK(BM_Conv2D_Batch_32);

static void BM_Conv2D_StridePad(benchmark::State& state) {
    RunConv2D(state, 1, 16, 64, 64, 16, 3, 3, 2, 2, 1, 1);
}

BENCHMARK(BM_Conv2D_StridePad);

static void RunConv3D(benchmark::State& state, size_t batch, size_t inChannels, size_t inD, size_t inH, size_t inW, size_t outChannels, size_t kD, size_t kH, size_t kW,
    size_t strideD = 1, size_t strideH = 1, size_t strideW = 1, size_t padD = 0,  size_t padH = 0, size_t padW = 0) {
    Tensor<float> input{ {batch, inChannels, inD, inH, inW} };
    Tensor<float> kernel{ {outChannels, inChannels, kD, kH, kW} };
    Tensor<float> bias{ {outChannels} };
    input.Fill(1.0f);
    kernel.Fill(0.01f);
    bias.Fill(0.0f);

    size_t checkpoint = MLCore::Runtime::MLContext::GetAllocator().Checkpoint();

    size_t outD = (inD + 2 * padD - kD) / strideD + 1;
    size_t outH = (inH + 2 * padH - kH) / strideH + 1;
    size_t outW = (inW + 2 * padW - kW) / strideW + 1;

    for (auto _ : state) {
        auto output = Conv3D(input, kernel, &bias, strideD, strideH, strideW, padD, padH, padW);
        benchmark::DoNotOptimize(output.Data());

        state.PauseTiming();
        MLCore::Runtime::MLContext::GetAllocator().RestoreCheckpoint(checkpoint);
        state.ResumeTiming();
    }

    int64_t flopsPerOutput = static_cast<int64_t>(inChannels * kD * kH * kW) * 2;
    int64_t numOutputs = static_cast<int64_t>(batch * outChannels * outD * outH * outW);
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * numOutputs * flopsPerOutput);
}

static void BM_Conv3D_EarlyLayer(benchmark::State& state) {
    RunConv3D(state, 1, 3, 8, 8, 8, 16, 3, 3, 3);
}

static void BM_Conv3D_DeepLayer(benchmark::State& state) {
    RunConv3D(state, 1, 32, 8, 8, 8, 32, 3, 3, 3);
}

BENCHMARK(BM_Conv3D_EarlyLayer);
BENCHMARK(BM_Conv3D_DeepLayer);

static void BM_Conv3D_Channels_4(benchmark::State& state) {
    RunConv3D(state, 1, 4, 8, 8, 8, 8, 3, 3, 3);
}

static void BM_Conv3D_Channels_16(benchmark::State& state) {
    RunConv3D(state, 1, 16, 8, 8, 8, 32, 3, 3, 3);
}

static void BM_Conv3D_Channels_64(benchmark::State& state) {
    RunConv3D(state, 1, 64, 8, 8, 8, 128, 3, 3, 3);
}

BENCHMARK(BM_Conv3D_Channels_4);
BENCHMARK(BM_Conv3D_Channels_16);
BENCHMARK(BM_Conv3D_Channels_64);

static void BM_Conv3D_Spatial_4(benchmark::State& state) {
    RunConv3D(state, 1, 16, 4, 4, 4, 16, 3, 3, 3);
}

static void BM_Conv3D_Spatial_16(benchmark::State& state) {
    RunConv3D(state, 1, 16, 16, 16, 16, 16, 3, 3, 3);
}

static void BM_Conv3D_Spatial_64(benchmark::State& state) {
    RunConv3D(state, 1, 16, 64, 64, 64, 16, 3, 3, 3);
}

BENCHMARK(BM_Conv3D_Spatial_4);
BENCHMARK(BM_Conv3D_Spatial_16);
BENCHMARK(BM_Conv3D_Spatial_64);

static void BM_Conv3D_Kernel_1x1(benchmark::State& state) {
    RunConv3D(state, 1, 32, 8, 8, 8, 32, 1, 1, 1);
}

static void BM_Conv3D_Kernel_3x3(benchmark::State& state) {
    RunConv3D(state, 1, 32, 8, 8, 8, 32, 3, 3, 3);
}

static void BM_Conv3D_Kernel_5x5(benchmark::State& state) {
    RunConv3D(state, 1, 32, 8, 8, 8, 32, 5, 5, 5);
}

BENCHMARK(BM_Conv3D_Kernel_1x1);
BENCHMARK(BM_Conv3D_Kernel_3x3);
BENCHMARK(BM_Conv3D_Kernel_5x5);

static void BM_Conv3D_Batch_1(benchmark::State& state) {
    RunConv3D(state, 1, 16, 8, 8, 8, 16, 3, 3, 3);
}

static void BM_Conv3D_Batch_4(benchmark::State& state) {
    RunConv3D(state, 4, 16, 8, 8, 8, 16, 3, 3, 3);
}

static void BM_Conv3D_Batch_16(benchmark::State& state) {
    RunConv3D(state, 16, 16, 8, 8, 8, 16, 3, 3, 3);
}

BENCHMARK(BM_Conv3D_Batch_1);
BENCHMARK(BM_Conv3D_Batch_4);
BENCHMARK(BM_Conv3D_Batch_16);

static void BM_Conv3D_StridePad(benchmark::State& state) {
    RunConv3D(state, 1, 16, 8, 8, 8, 16, 3, 3, 3, 2, 2, 2, 1, 1, 1);
}

BENCHMARK(BM_Conv3D_StridePad);