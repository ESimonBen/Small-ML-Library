// main.cpp
#include <iostream>
#include <mlCore/optimizers/sgd.h>
#include <mlCore/optimizers/adam.h>
#include <mlCore/operations/operations.h>

using namespace MLCore;
using namespace MLCore::Memory;
using namespace MLCore::AutoGrad;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;
using namespace MLCore::Optimizers;

int main() {
    ArenaAllocator allocator;

    Tensor<float> weight{ {1}, allocator };
    weight[0] = 0.0f;
    weight.SetRequiresGrad(true);

    Tensor<float> input{ {1}, allocator };
    input[0] = 2.0f;

    Tensor<float> target{ {1}, allocator };
    target[0] = 4.0f;

    std::vector<Parameter<float>> params{ Parameter<float>{weight} };
    Adam<float> optimizer{ params, .1f };

    for (int epoch = 0; epoch < 20; ++epoch) {
        // Forward: prediction = weight * input;
        auto predict = Multiply(weight, input, allocator);

        // Loss: mean squared error
        auto loss = MeanSquaredError(predict, target, allocator);

        std::cout << "Epoch " << epoch << " | Loss: " << loss[0] << " | w: " << weight[0] << std::endl;

        optimizer.ZeroGrad();
        loss.Backward();

        auto grad = weight.Grad();
        std::cout << "Grad: " << grad[0] << std::endl;

        optimizer.Step();
    }

	return 0;
}