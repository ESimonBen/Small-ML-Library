/// dummyGradFn.h
#pragma once
#include <mlCore/tensor/tensor.h>
#include <mlCore/autograd/gradientFn.h>

using namespace MLCore::AutoGrad;

template <typename T>
class DummyGradFn : public GradFn<T> {
public:
    using Impl = MLCore::TensorCore::TensorImpl<T>;

    DummyGradFn(std::shared_ptr<Impl> impl)
        : GradFn<T>(std::move(impl))
    {}

    DummyGradFn(std::vector<std::shared_ptr<Impl>> impls)
        : GradFn<T>(std::move(impls))
    {}

    void Backward(const MLCore::TensorCore::Tensor<T>&, MLCore::Memory::ArenaAllocator&) override{
        /// No implementation here
    }

    auto GetInput(size_t i) {
        return this->Input(i);
    }

    auto GetInputs() {
        return this->Inputs();
    }
};