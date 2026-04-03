// main.cpp
#include <iostream>
#include <mlCore/tensor/tensor.h>

using namespace MLCore;

int main() {
    Memory::ArenaAllocator allocator;
    TensorCore::Tensor<float> A({ 2, 3 }, allocator);

    A.Fill(0.0f);
    float num = 1.0f;

    A[5] = 1.0f;

    for (auto& v : A) {
        std::cout << v << " ";
    }

    A(0, 0) = 3.4f;
    A(1, 2) = 4.3f;

    for (auto& v : A) {
        std::cout << v << " ";
    }

    std::cout << std::endl;
    std::cout << std::endl << A.NumElements() << std::endl;
}