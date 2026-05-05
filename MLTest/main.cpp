// main.cpp
#include <iostream>
#include <mlCore/optimizers/adam.h>
#include <mlCore/optimizers/sgd.h>
#include <mlCore/operations/operations.h>
#include <mlCore/module/layers/linearLayer.h>
#include <mlCore/module/sequential.h>
#include <mlCore/module/layers/tanhLayer.h>
#include <mlCore/module/layers/reluLayer.h>
#include <mlCore/module/layers/leakyReluLayer.h>
#include <mlCore/module/layers/sigmoidLayer.h>
#include <mlCore/training/trainer.h>

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
    std::cout << "Test With Adam (No Sigmoid + BCE With Logits)" << std::endl;
    std::cout << "=== XOR Nonlinear Test ===\n";

    // -----------------------------
    // 1. Dataset (XOR)
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
    model.Emplace<SigmoidLayer<float>>();

    // Collect parameters
    auto params = model.GetParameters();

    SGD<float> opt{ params, 0.1f };

    // -----------------------------
    // 3. Training loop
    // -----------------------------

    Trainer<float> trainer{ model, opt,
        [&](const auto& pred, const auto& target) {
            return BinaryCrossEntropy(pred, target, Reduction::Mean, allocator);
        }
    };

    trainer.OnEpochEnd = [](int epoch, float loss) {
        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch
            << " | Loss: " << loss << std::endl;
        }
    };

    trainer.Fit(x, y, 20000);
}

int main() {
    ArenaAllocator allocator;

    TestXOR(allocator);

    return 0;
}