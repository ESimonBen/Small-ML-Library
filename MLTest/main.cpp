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

using namespace MLCore;
using namespace MLCore::Memory;
using namespace MLCore::AutoGrad;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;
using namespace MLCore::Optimizers;
using namespace MLCore::NN;
using namespace MLCore::Init;


void TestXOR(ArenaAllocator& allocator) {
    std::cout << "Test With Adam" << std::endl;
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

    /*Adam<float> opt{ params, 0.01f };*/
    SGD<float> opt{ params, 0.1f };

    // -----------------------------
    // 3. Training loop
    // -----------------------------
    for (int epoch = 0; epoch < 20000; ++epoch) {

        // Forward
        auto pred = model(x);

        // Loss
        auto loss = BinaryCrossEntropy(pred, y, Reduction::Mean, allocator);

        // Backprop
        opt.ZeroGrad();
        loss.Backward();
        opt.Step();

        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch
                << " | Loss: " << loss[0] << std::endl;
        }
    }

    // -----------------------------
    // 4. Evaluation
    // -----------------------------
    auto pred = model(x);

    size_t correct = 0;

    for (size_t i = 0; i < 4; ++i) {
        float prob = pred[i];

        int pred = (prob > 0.5f) ? 1 : 0;
        int actual = (y[i] > 0.5f) ? 1 : 0;

        if (pred == actual)
            correct++;
    }

    std::cout << "Accuracy: " << (float)correct / 4 << std::endl;

    std::cout << "Final logits:\n";
    for (size_t i = 0; i < 4; ++i) {
        std::cout << pred[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    ArenaAllocator allocator;

    TestXOR(allocator);

    return 0;
}