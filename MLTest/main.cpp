// main.cpp
#include <iostream>
#include <mlCore/optimizers/sgd.h>
#include <mlCore/optimizers/adam.h>
#include <mlCore/training/trainer.h>
#include <mlCore/module/sequential.h>
#include <mlCore/module/layers/layers.h>
#include <mlCore/operations/operations.h>

using namespace MLCore;
using namespace MLCore::Memory;
using namespace MLCore::AutoGrad;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;
using namespace MLCore::Optimizers;
using namespace MLCore::NN;
using namespace MLCore::Init;
using namespace MLCore::Training;


void TestXOR(ArenaAllocator& allocator) {
    std::cout << "=== XOR Linear Test ===\n";

    // -----------------------------
    // 1. Dataset (AND)
    // -----------------------------
    Tensor<float> x{ {4, 2}, allocator };
    Tensor<float> y{ {4, 1}, allocator };

    // (0,0) -> 0
    x[0] = 0; 
    x[1] = 0; 
    y[0] = 0;

    // (0,1) -> 1
    x[2] = 0; 
    x[3] = 1; 
    y[1] = 1;

    // (1,0) -> 1
    x[4] = 1; 
    x[5] = 0; 
    y[2] = 1;

    // (1,1) -> 0
    x[6] = 1; 
    x[7] = 1;
    y[3] = 0;

    // -----------------------------
    // 2. Model (3-layer MLP)
    // -----------------------------
    Sequential<float> model;

    model.Emplace<LinearLayer<float>>(2, 8, allocator, InitType::HeUniform);
    model.Emplace<LeakyReLULayer<float>>();
    model.Emplace<LinearLayer<float>>(8, 1, allocator, InitType::HeUniform);

    // Collect parameters
    auto params = model.GetParameters();

    SGDMomentum<float> opt{ params, 0.1f, 0.1f };

    // -----------------------------
    // 3. Training loop
    // -----------------------------

    Trainer<float> trainer{ model, opt,
        [&](const auto& pred, const auto& target) {
            return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean, allocator);
        }
    };

    trainer.AddMetric("Accuracy", [](const TensorCore::Tensor<float>& pred, const TensorCore::Tensor<float>& target) -> float {
        size_t correct = 0;
        size_t size = pred.NumElements();

        for (size_t i = 0; i < size; ++i) {
            int predict = (pred[i] > 0) ? 1 : 0;

            if (predict == static_cast<int>(target[i])) {
                correct++;
            }
        }

        return static_cast<float>(correct) / size;
    });

    trainer.OnEpochEnd =
        [](const EpochStats<float>& stats) {

        if (stats.epoch % 500 == 0) {
            std::cout
                << "Epoch " << stats.epoch
                << " | Train Loss: " << stats.trainLoss
                << " | Train Accuracy: "
                << (stats.trainMetrics.at("Accuracy") * 100) << "%"
                << " | Val Loss: "
                << stats.valLoss
                << " | Val Accuracy: "
                << (stats.valMetrics.at("Accuracy") * 100) << "%"
                << '\n';
        }
    };

    trainer.Fit(x, y, x, y, 10000, 4);
}

int main() {
    ArenaAllocator allocator;

    TestXOR(allocator);

    return 0;
}