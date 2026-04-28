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

int main() {
    ArenaAllocator allocator;

    // Two parameters (simulate different layers)
    Tensor<float> w1{ {1}, allocator };
    Tensor<float> w2{ {1}, allocator };

    w1[0] = 0.0f;
    w2[0] = 0.0f;

    w1.SetRequiresGrad(true);
    w2.SetRequiresGrad(true);

    Tensor<float> input{ {1}, allocator };
    input[0] = 2.0f;

    Tensor<float> target{ {1}, allocator };
    target[0] = 4.0f;

    // Wrap parameters
    Parameter<float> p1{ w1 };
    Parameter<float> p2{ w2 };

    std::vector<Parameter<float>> params1{ p1 };
    std::vector<Parameter<float>> params2{ p2 };

    // Two parameter groups with different LRs
    ParameterGroup<float> group1{ params1, 0.1f };   // fast
    ParameterGroup<float> group2{ params2, 0.01f };  // slow

    std::vector<ParameterGroup<float>> groups{ group1, group2 };

    SGDMomentum<float> optimizer{ groups, .1f};

    // Scheduler (decays every 5 steps)
    ExponentialLR<float> scheduler{ optimizer, 0.5f };

    for (int epoch = 0; epoch < 100; ++epoch) {
        // Forward: (w1 + w2) * input
        auto sum = Add(w1, w2, allocator);
        auto predict = Multiply(sum, input, allocator);

        auto loss = MeanSquaredError(predict, target, allocator);

        std::cout << "Epoch " << epoch
            << " | Loss: " << loss[0]
            << " | w1: " << w1[0]
            << " | w2: " << w2[0]
            << std::endl;

        optimizer.ZeroGrad();
        loss.Backward();

        optimizer.Step();
        scheduler.UpdateLR();

        // Debug: print group learning rates
        auto& groupsRef = optimizer.ParamGroups();
        std::cout << "  LR group1: " << groupsRef[0].learningRate
            << " | LR group2: " << groupsRef[1].learningRate
            << std::endl;
    }

    return 0;
}