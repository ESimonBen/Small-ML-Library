// main.cpp
#include <iostream>
#include <mlCore/optimizers/adam.h>
#include <mlCore/optimizers/sgd.h>
#include <mlCore/operations/operations.h>
#include <mlCore/module/linearLayer.h>

using namespace MLCore;
using namespace MLCore::Memory;
using namespace MLCore::AutoGrad;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;
using namespace MLCore::Optimizers;
using namespace MLCore::NN;


void TestXOR(ArenaAllocator& allocator) {
    std::cout << "Test With SGD (More Iteration)" << std::endl;
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
    // 2. Model (2-layer MLP)
    // -----------------------------
    LinearLayer<float> l1(2, 4, allocator);
    LinearLayer<float> l2(4, 1, allocator);

    // Collect parameters manually
    auto p1 = l1.GetParameters();
    auto p2 = l2.GetParameters();

    std::vector<std::reference_wrapper<NN::Parameter<float>>> params;
    params.insert(params.end(), p1.begin(), p1.end());
    params.insert(params.end(), p2.begin(), p2.end());

    /*Adam<float> opt{ params, 0.01f };*/
    SGD<float> opt{ params, 0.1f };

    // -----------------------------
    // 3. Training loop
    // -----------------------------
    for (int epoch = 0; epoch < 20000; ++epoch) {

        // Forward
        auto h = Tanh(l1.Forward(x), allocator);
        auto logits = l2.Forward(h);

        // Loss
        auto loss = BinaryCrossEntropyWithLogits(logits, y, Reduction::Mean, allocator);

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
    auto h = Tanh(l1.Forward(x), allocator);
    auto logits = l2.Forward(h);

    size_t correct = 0;

    for (size_t i = 0; i < 4; ++i) {
        float logit = logits[i];
        float prob = 1.0f / (1.0f + std::exp(-logit));

        int pred = (prob > 0.5f) ? 1 : 0;
        int actual = (y[i] > 0.5f) ? 1 : 0;

        if (pred == actual)
            correct++;
    }

    std::cout << "Accuracy: " << (float)correct / 4 << std::endl;

    std::cout << "Final logits:\n";
    for (size_t i = 0; i < 4; ++i) {
        std::cout << logits[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    ArenaAllocator allocator;

    TestXOR(allocator);

    return 0;
}
