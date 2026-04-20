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

    auto result = AxisSum(predict, 0, allocator);

    for (auto& v : result) {
        std::cout << v << " ";
    }
    std::cout << std::endl << std::endl;

    Tensor<float> gradient{ {2}, allocator };
    gradient.Fill(1.0f);

    result.Backward(gradient);

    auto grad1 = predict.Grad();

    std::cout << "Add Gradient: Grad1" << std::endl;
    for (auto& v : grad1) {
        std::cout << v << " ";
    }

	return 0;
}