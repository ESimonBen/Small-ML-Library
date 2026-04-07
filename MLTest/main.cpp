// main.cpp
#include <iostream>
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>
#include <mlCore/operations/elementwise/elementwise.h>
#include <mlCore/autograd/functions/elementwise/addGradFn.h>
#include <mlCore/autograd/functions/elementwise/subGradFn.h>
#include <mlCore/autograd/functions/elementwise/mulGradFn.h>
#include <mlCore/autograd/functions/elementwise/divGradFn.h>
#include <mlCore/autograd/functions/elementwise/negateGradFn.h>
#include <mlCore/autograd/functions/elementwise/powerGradFn.h>

using namespace MLCore;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;
using namespace MLCore::Memory;
using namespace MLCore::AutoGrad;

int main() {
    ArenaAllocator allocator;

    // 2x2 tensors for easy verification
    Tensor<float> a({ 2, 2 }, allocator);
    Tensor<float> b({ 2, 2 }, allocator);

    // Fill with some values
    a[0] = 1.0f; a[1] = 2.0f; a[2] = 3.0f; a[3] = 4.0f;
    b[0] = 5.0f; b[1] = 6.0f; b[2] = 7.0f; b[3] = 8.0f;

    // Make them require gradients
    a.SetRequiresGrad(true);
    b.SetRequiresGrad(true);

    // Grad output (usually comes from loss)
    Tensor<float> gradOut({ 2,2 }, allocator);
    gradOut[0] = 1.0f; gradOut[1] = 1.0f; gradOut[2] = 1.0f; gradOut[3] = 1.0f;

    AddGradFn<float> addFn(&a, &b);
    addFn.Backward(gradOut);

    std::cout << "Add Gradient - a:\n";
    for (auto v : *a.Grad()) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";

    std::cout << "Add Gradient - b:\n";
    for (auto v : *b.Grad()) {
        std::cout << v << " ";
    }

    std::cout << "\n\n";

    a.ZeroGrad();
    b.ZeroGrad();

    SubGradFn<float> subFn(&a, &b);
    subFn.Backward(gradOut);

    std::cout << "Sub Gradient - a:\n";
    for (auto v : *a.Grad()) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";

    std::cout << "Sub Gradient - b:\n";
    for (auto v : *b.Grad()) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";

    a.ZeroGrad();
    b.ZeroGrad();

    MulGradFn<float> mulFn(&a, &b);
    mulFn.Backward(gradOut);

    std::cout << "Mul Gradient - a:\n";
    for (auto v : *a.Grad()) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";

    std::cout << "Mul Gradient - b:\n";
    for (auto v : *b.Grad()) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";

    a.ZeroGrad();
    b.ZeroGrad();

    DivGradFn<float> divFn(&a, &b);
    divFn.Backward(gradOut);

    std::cout << "Div Gradient - a:\n";
    for (auto v : *a.Grad()) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";

    std::cout << "Div Gradient - b:\n";
    for (auto v : *b.Grad()) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";

    Tensor<float> c({ 1,2 }, allocator); // shape (1,2)
    Tensor<float> d({ 2,2 }, allocator); // shape (2,2)

    c[0] = 1; c[1] = 2;
    d[0] = 3; d[1] = 4; d[2] = 5; d[3] = 6;

    c.SetRequiresGrad(true);
    d.SetRequiresGrad(true);

    AddGradFn<float> addBroadcast(&c, &d);
    addBroadcast.Backward(gradOut); // gradOut must match broadcasted shape

    std::cout << "Broadcast Add Gradient - c:\n";
    for (auto v : *c.Grad()) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";

    std::cout << "Broadcast Add Gradient - d:\n";
    for (auto v : *d.Grad()) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";

    c.ZeroGrad();
    d.ZeroGrad();

    SubGradFn<float> subBroadcast(&c, &d);
    subBroadcast.Backward(gradOut);

    std::cout << "Broadcast Sub Gradient - c:\n";
    for (auto v : *c.Grad()) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";

    std::cout << "Broadcast Sub Gradient - d:\n";
    for (auto v : *d.Grad()) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";

    c.ZeroGrad();
    d.ZeroGrad();

    MulGradFn<float> mulBroadcast(&c, &d);
    mulBroadcast.Backward(gradOut);

    std::cout << "Broadcast Mul Gradient - c:\n";
    for (auto v : *c.Grad()) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";

    std::cout << "Broadcast Mul Gradient - d:\n";
    for (auto v : *d.Grad()) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";

    c.ZeroGrad();
    d.ZeroGrad();

    DivGradFn<float> divBroadcast(&c, &d);
    divBroadcast.Backward(gradOut);

    std::cout << "Broadcast Div Gradient - c:\n";
    for (auto v : *c.Grad()) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";

    std::cout << "Broadcast Div Gradient - d:\n";
    for (auto v : *d.Grad()) {
        std::cout << v << " ";
    }
    std::cout << "\n\n";

    return 0;
}