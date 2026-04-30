# Small ML Library
A small, educational machine learning library built in C++.

## Overview

This library provides:

-   A custom **Tensor** implementation with shape handling and broadcasting
-   A modular **autograd engine** for gradient computation
-   A structured set of **mathematical operations** (elementwise, reduction, linear algebra, etc.)
-   Built-in **optimizers** and **learning rate schedulers**
-   A clean separation between forward operations and backward gradient logic

This project is heavily inspired by the internal design philosophies of modern frameworks, but implemented from the ground up for learning and control.

* * *

## Core Features

### Tensor System

-   Multi-dimensional tensor support
-   Shape utilities and broadcasting semantics
-   Efficient memory handling via custom allocator and storage system

### Automatic Differentiation (Autograd)

-   Backpropagation through computational graphs
-   Modular gradient function system (`gradientFn`)
-   Operation-specific gradient implementations:
    -   Elementwise
    -   Linear algebra
    -   Reductions
    -   Activations
    -   Loss functions
    -   Scalars

### Operations

Organized by category:

-   **Elementwise**: add, multiply, etc.
-   **Linear Algebra**: matrix operations
-   **Reduction**: sum, mean, max, min
-   **Broadcasting**: shape alignment and expansion
-   **Activations**: nonlinear functions
-   **Loss**: training objectives

### Optimizers

-   SGD (Stochastic Gradient Descent)
-   Adam

### Learning Rate Schedulers

-   StepLR
-   ExponentialLR

* * *

## Architecture

The library is structured to clearly separate responsibilities:

    MLCore/  
    autograd/        # Gradient system and backpropagation logic  
    operations/      # Forward operation implementations  
    tensor/          # Core tensor abstraction  
    memory/          # Memory management (allocator, storage)  
    optimizers/      # Optimization algorithms  
    schedulers/      # Learning rate scheduling  
    utils/           # Shape and utility functions

### Key Design Ideas

-   **Separation of forward and backward logic**  
    Each operation has a corresponding gradient function, making the system extensible and easier to debug.
-   **Header + inline implementation (`.h` + `.inl`)**  
    Enables performance while maintaining readability.
-   **Modular operation categories**  
    Keeps the system scalable as more functionality is added.

* * *

## Build Instructions

### Requirements

-   C++17 or later
-   CMake

### Build

    git clone https://github.com/ESimonBen/Small-ML-Library.git
    cd Small-ML-Library
    cmake -S . -B build -G "Visual Studio 17 2022" -A x64 # Visual Studio build example, you may modify for your specific system

### Run Tests

    cd bin       # Binaries are generated into this bin directory
    ./MLTest.exe # Windows example

* * *

##  Example (Conceptual)

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

    size_t batchSize = 4;

    // Parameters
    Tensor<float> w1{ {1}, allocator };
    Tensor<float> w2{ {1}, allocator };

    w1[0] = 0.0f;
    w2[0] = 0.0f;

    w1.SetRequiresGrad(true);
    w2.SetRequiresGrad(true);

    // Batched input
    Tensor<float> input{ {batchSize, 1}, allocator };
    Tensor<float> target{ {batchSize, 1}, allocator };

    for (size_t i = 0; i < batchSize; ++i) {
        input[i] = static_cast<float>(i + 1);   // 1,2,3,4
        target[i] = 2.0f * input[i];            // ideal: y = 2x
    }

    // Parameters
    Parameter<float> p1{ w1 };
    Parameter<float> p2{ w2 };

    std::vector<Parameter<float>> params1{ p1 };
    std::vector<Parameter<float>> params2{ p2 };

    ParameterGroup<float> group1{ params1, 0.1f };
    ParameterGroup<float> group2{ params2, 0.01f };

    SGD<float> optimizer{ {group1, group2} };
    ExponentialLR<float> scheduler{ optimizer, 0.99f };

    for (int epoch = 0; epoch < 100; ++epoch) {

        // Forward
        auto sum = Add(w1, w2, allocator);                  // scalar
        auto predict = Multiply(sum, input, allocator);     // broadcast over batch

        auto loss = MeanSquaredError(
            predict,
            target,
            Reduction::Mean,
            allocator
        );

        std::cout << "Epoch " << epoch
            << " | Loss: " << loss[0]
            << " | w1: " << w1[0]
            << " | w2: " << w2[0]
            << std::endl;

        optimizer.ZeroGrad();
        loss.Backward();
        optimizer.Step();
        scheduler.UpdateLR();
    }

    return 0;
}

* * *

## Project Goals

This project is focused on:

-   Understanding how **autograd systems** work internally
-   Implementing **broadcasting and tensor operations correctly**
-   Building a foundation for **training neural networks from scratch**
-   Gaining low-level insight into ML frameworks

* * *

## Current Status

This library is actively under development.

Implemented / In Progress:

-   Tensor operations and broadcasting
-   Autograd framework
-   Core ops (elementwise, reduction, linear algebra)
-   Optimizers and schedulers

Planned:

-   Expanded operation coverage
-   Improved numerical stability
-   End-to-end neural network examples
-   Performance optimizations

* * *

## Why This Project Matters

Most ML users rely on high-level APIs. This library focuses on what happens _under the hood_:

-   How gradients are actually computed
-   How broadcasting works across dimensions
-   How optimizers update parameters

This makes it a strong educational and systems-focused project.

* * *

## Contributing

This is primarily a personal learning project, but suggestions and discussions are welcome.

* * *

## License

MIT License

Copyright (c) 2026 ESimonBen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


* * *
