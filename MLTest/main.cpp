// main.cpp
#include <iostream>
#include <random>
#include <mlCore/operations/elementwise/elementwise.h>
#include <mlCore/operations/scalar/scalar.h>
#include <mlCore/operations/reduction/reduction.h>
#include <mlCore/operations/linearAlgebra/linalg.h>

using namespace MLCore;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;
using namespace MLCore::Memory;

int main() {
    ArenaAllocator allocator;
    Tensor<float> A{ {3, 4, 5}, allocator };
    // Tensor<float> vec2{ {50}, allocator };

    std::random_device rd; // Seed
    std::mt19937 gen(rd()); // Engine
    std::uniform_int_distribution<> distr(1, 100); // Range [1, 100]

    //mat2.Fill(1.0f);

    for (size_t i = 0; i < A.NumElements(); ++i) {
        A[i] = distr(gen);
        A[i];
    }

    std::cout << "Original: " << std::endl;
    for (size_t x = 0; x < 3; ++x) {
        for (size_t y = 0; y < 4; ++y) {
            for (size_t z = 0; z < 5; ++z) {
                std::cout << A(x, y, z) << " ";
            }
            
            std::cout << std::endl;
        }
        
        if (x < 2) {
            std::cout << "|          |";
        }

        std::cout << std::endl;
    }

    auto sum = AxisSum(A, 2, allocator);
    auto dimensions = sum.Dims();
    std::cout << "Sum Dimensions: ";
    for (size_t i = 0; i < dimensions.size(); ++i) {
        std::cout << dimensions[i] << " ";
    }

    return 0;
}