// main.cpp
#include <iostream>
#include <mlCore/operations/loss/loss.h>
#include <mlCore/operations/reduction/reduction.h>
#include <mlCore/operations/linearAlgebra/linalg.h>
#include <mlCore/operations/activations/activation.h>
#include <mlCore/operations/elementwise/elementwise.h>

using namespace MLCore;
using namespace MLCore::Memory;
using namespace MLCore::AutoGrad;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

int main() {
    ArenaAllocator allocator;

    Tensor<float> predict({ 2, 2 }, allocator);
    predict.Fill(1.0f);
    predict[0] = 4.0f;
    predict.SetRequiresGrad(true);

    Tensor<float> target({ 2, 2 }, allocator);
    target.Fill(0.0f);
    target[0] = 1.0f;

    auto result = CrossEntropyWithLogits(predict, target, 0, allocator);

    for (auto& v : result) {
        std::cout << v << " ";
    }

    std::cout << std::endl << std::endl;

    result.Backward();

    auto grad1 = predict.Grad();

    std::cout << "Add Gradient: Grad1" << std::endl;
    for (auto& v : grad1) {
        std::cout << v << " ";
    }

	return 0;
}

/*Tensor<float> gradient{ {2}, allocator };
    gradient.Fill(1.0f);*/