// main.cpp
#include <iostream>
#include <mlCore/tensor/tensor.h>
#include <mlCore/operations/elementwise/elementwise.h>
#include <mlCore/memory/allocator.h>

using namespace MLCore;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;
using namespace MLCore::Memory;

template <typename T>
void PrintTensor(const Tensor<T>& tensor, const std::string& name) {
    std::cout << name << " (shape: ";
    for (size_t i = 0; i < tensor.GetShape().Rank(); ++i) {
        std::cout << tensor.GetShape()[i];
        if (i + 1 < tensor.GetShape().Rank()) std::cout << "x";
    }
    std::cout << "): ";

    for (size_t i = 0; i < tensor.NumElements(); ++i) {
        std::cout << tensor.Data()[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    ArenaAllocator allocator;

    // ----------------------
    // 1. Same shape test
    // ----------------------
    Tensor<float> A({ 2, 3 }, allocator);
    Tensor<float> B({ 2, 3 }, allocator);

    for (size_t i = 0; i < A.NumElements(); ++i) {
        A.Data()[i] = static_cast<float>(i + 1);
        B.Data()[i] = static_cast<float>((i + 1) * 10);
    }

    auto C_add = Add(A, B, allocator);
    auto C_sub = Subtract(A, B, allocator);
    auto C_mul = Multiply(A, B, allocator);
    auto C_div = Divide(B, A, allocator);

    PrintTensor(A, "A");
    PrintTensor(B, "B");
    PrintTensor(C_add, "A + B");
    PrintTensor(C_sub, "A - B");
    PrintTensor(C_mul, "A * B");
    PrintTensor(C_div, "B / A");

    // ----------------------
    // 2. Broadcast last dimension
    // ----------------------
    Tensor<float> D({ 2, 1 }, allocator);
    Tensor<float> E({ 2, 3 }, allocator);

    for (size_t i = 0; i < D.NumElements(); ++i) D.Data()[i] = 1.0f;
    for (size_t i = 0; i < E.NumElements(); ++i) E.Data()[i] = static_cast<float>(i + 1);

    auto F_add = Add(D, E, allocator);
    PrintTensor(D, "D");
    PrintTensor(E, "E");
    PrintTensor(F_add, "D + E");

    // ----------------------
    // 3. Broadcast first dimension
    // ----------------------
    Tensor<float> G({ 1, 3 }, allocator);
    Tensor<float> H({ 2, 3 }, allocator);

    for (size_t i = 0; i < G.NumElements(); ++i) G.Data()[i] = static_cast<float>(i + 10);
    for (size_t i = 0; i < H.NumElements(); ++i) H.Data()[i] = static_cast<float>(i + 1);

    auto I_mul = Multiply(G, H, allocator);
    PrintTensor(G, "G");
    PrintTensor(H, "H");
    PrintTensor(I_mul, "G * H");

    // ----------------------
    // 4. Scalar broadcasting
    // ----------------------
    Tensor<float> scalar({ 1 }, allocator);
    Tensor<float> J({ 2, 2 }, allocator);

    scalar.Data()[0] = 5.0f;
    for (size_t i = 0; i < J.NumElements(); ++i) J.Data()[i] = static_cast<float>(i + 1);

    auto K_add = Add(scalar, J, allocator);
    PrintTensor(scalar, "Scalar");
    PrintTensor(J, "J");
    PrintTensor(K_add, "Scalar + J");

    return 0;
}