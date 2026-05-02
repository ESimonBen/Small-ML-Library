// main.cpp
#include <iostream>
#include <mlCore/optimizers/adam.h>
#include <mlCore/optimizers/sgd.h>
#include <mlCore/schedulers/stepLR.h>
#include <mlCore/schedulers/expLR.h>
#include <mlCore/operations/operations.h>

using namespace MLCore;
using namespace MLCore::Memory;
using namespace MLCore::AutoGrad;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;
using namespace MLCore::Optimizers;
using namespace MLCore::Schedulers;
using namespace MLCore::NN;


void TestMultiFeature(ArenaAllocator& allocator) {
    std::cout << "\n=== Test: Multi-feature batch ===\n";

    size_t B = 4;
    size_t F = 3;

    Tensor<float> input{ {B, F}, allocator };
    Tensor<float> weights{ {F, 1}, allocator };
    weights.Fill(0.0f);

    weights.SetRequiresGrad(true);

    // Fill input
    for (size_t i = 0; i < B * F; ++i)
        input[i] = static_cast<float>(i + 1);

    // Target = sum of features * 2
    Tensor<float> target{ {B, 1}, allocator };
    for (size_t b = 0; b < B; ++b) {
        float sum = 0;
        for (size_t f = 0; f < F; ++f)
            sum += input[b * F + f];
        target[b] = 2.0f * sum;
    }


    // Forward
    auto pred = MatMultiply(input, weights, allocator);
    std::cout << "Predictions:\n";
    for (auto& v : pred) std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "Targets:\n";
    for (auto& v : target) std::cout << v << " ";
    std::cout << std::endl;

    auto loss = MeanSquaredError(pred, target, Reduction::Mean, allocator);

    std::cout << "Loss: " << loss[0] << std::endl;

    loss.Backward();

    std::cout << "Gradients:\n";
    for (auto& v : weights.Grad())
        std::cout << v << " ";
    std::cout << std::endl;
}

void TestSoftmaxCrossEntropy(ArenaAllocator& allocator) {
    std::cout << "\n=== Test: Softmax + CrossEntropy ===\n";

    size_t B = 3;
    size_t C = 4;

    Tensor<float> logits{ {B, C}, allocator };
    Tensor<float> targets{ {B, C}, allocator };

    logits.SetRequiresGrad(true);

    // Fill logits
    for (size_t i = 0; i < B * C; ++i)
        logits[i] = static_cast<float>(static_cast<int>(i % C) - 1);

    // One-hot targets
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < C; ++c) {
            targets[b * C + c] = (c == (b % C)) ? 1.0f : 0.0f;
        }
    }

    auto loss = CrossEntropyWithLogits(logits, targets, Reduction::Mean, allocator);

    std::cout << "Loss: " << loss[0] << std::endl;

    loss.Backward();

    std::cout << "Gradient sample: " << logits.Grad()[0] << std::endl;
}

void TestGradientCheck(ArenaAllocator& allocator) {
    std::cout << "\n=== Test: Gradient Check ===\n";

    float epsilon = 1e-4f;

    Tensor<float> w{ {1}, allocator };
    w[0] = 1.5f;
    w.SetRequiresGrad(true);

    Tensor<float> x{ {1}, allocator };
    x[0] = 3.0f;

    Tensor<float> y{ {1}, allocator };
    y[0] = 6.0f;

    // Forward
    auto pred = Multiply(w, x, allocator);
    auto loss = MeanSquaredError(pred, y, Reduction::Mean, allocator);

    loss.Backward();

    float autogradGrad = w.Grad()[0];

    // Numerical gradient
    w[0] += epsilon;
    auto loss1 = MeanSquaredError(Multiply(w, x, allocator), y, Reduction::Mean, allocator);

    w[0] -= 2 * epsilon;
    auto loss2 = MeanSquaredError(Multiply(w, x, allocator), y, Reduction::Mean, allocator);

    float numericalGrad = (loss1[0] - loss2[0]) / (2 * epsilon);

    std::cout << "Autograd:   " << autogradGrad << std::endl;
    std::cout << "Numerical:  " << numericalGrad << std::endl;
}

void TestBalancedLR(ArenaAllocator& allocator) {
    std::cout << "\n=== Test: Balanced LR ===\n";

    Tensor<float> w1{ {1}, allocator };
    Tensor<float> w2{ {1}, allocator };

    w1[0] = 0.0f;
    w2[0] = 0.0f;

    w1.SetRequiresGrad(true);
    w2.SetRequiresGrad(true);

    Tensor<float> x{ {1}, allocator };
    x[0] = 2.0f;

    Tensor<float> y{ {1}, allocator };
    y[0] = 4.0f;

    Parameter<float> p1{ w1 };
    Parameter<float> p2{ w2 };

    ParameterGroup<float> g1{ {p1}, 0.1f };
    ParameterGroup<float> g2{ {p2}, 0.1f };

    SGD<float> opt{ {g1, g2} };

    for (int i = 0; i < 50; ++i) {
        auto sum = Add(w1, w2, allocator);
        auto pred = Multiply(sum, x, allocator);
        auto loss = MeanSquaredError(pred, y, Reduction::Mean, allocator);

        opt.ZeroGrad();
        loss.Backward();
        opt.Step();
    }

    std::cout << "w1: " << w1[0] << " | w2: " << w2[0] << std::endl;
}

int main() {
    ArenaAllocator allocator;

    TestMultiFeature(allocator); // failed
    TestSoftmaxCrossEntropy(allocator); // failed
    TestGradientCheck(allocator);
    TestBalancedLR(allocator);

    return 0;
}
