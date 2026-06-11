// main.cpp
#include <iostream>
#include <mlCore/optimizers/sgd.h>
#include <mlCore/optimizers/adam.h>
#include <mlCore/training/trainer.h>
#include <mlCore/schedulers/stepLR.h>
#include <mlCore/module/sequential.h>
#include <mlCore/module/layers/layers.h>
#include <mlCore/operations/operations.h>
#include <mlCore/serialization/checkpoint.h>

using namespace MLCore;
using namespace MLCore::Memory;
using namespace MLCore::AutoGrad;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;
using namespace MLCore::Optimizers;
using namespace MLCore::NN;
using namespace MLCore::Init;
using namespace MLCore::Training;
using namespace MLCore::Schedulers;
using namespace MLCore::Serialization;

void TestXOR(ArenaAllocator& allocator) {
    std::cout << "=== XOR Nonlinear Test ===\n";

    // -----------------------------
    // 1. Dataset (AND)
    // -----------------------------
    Tensor<float> x{ {4, 2}, allocator };
    Tensor<float> y{ {4, 1}, allocator };

    // (0,0) -> 0
    x[0] = 0; 
    x[1] = 0; 
    y[0] = 0;

    // (0,1) -> 1
    x[2] = 0; 
    x[3] = 1; 
    y[1] = 1;

    // (1,0) -> 1
    x[4] = 1; 
    x[5] = 0; 
    y[2] = 1;

    // (1,1) -> 0
    x[6] = 1; 
    x[7] = 1;
    y[3] = 0;

    // -----------------------------
    // 2. Model (3-layer MLP)
    // -----------------------------
    Sequential<float> model;

    model.EmplaceNamed<LinearLayer<float>>("layer1", 2, 8, allocator, InitType::HeUniform);
    model.EmplaceNamed<LeakyReLULayer<float>>("leakyReLU");
    model.EmplaceNamed<LinearLayer<float>>("layer2", 8, 1, allocator, InitType::HeUniform);

    // Collect parameters
    auto params = model.GetParameters();
    auto namedParams = model.GetNamedParameters();

    for (auto& [name, p] : namedParams) {
        std::cout << "Name: " << name << std::endl;
        std::cout << "Values: " << std::endl;

        auto& tensor = p.get().Data();
        auto size = tensor.NumElements();

        for (size_t i = 0; i < size; ++i) {
            std::cout << tensor[i] << " ";

            if (i == size - 1) {
                std::cout << std::endl;
            }
        }
    }

    SGDMomentum<float> opt{ params, 0.1f, 0.1f };

    StepLR<float> scheduler{ opt, 1000, .99f };

    // -----------------------------
    // 3. Training loop
    // -----------------------------
    Trainer<float> trainer{ model, opt,
        [&](const auto& pred, const auto& target) {
            return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean, allocator);
        }
    };

    trainer.SetScheduler(scheduler, SchedulerStepMode::Epoch);

    trainer.AddMetric("Accuracy", [](const TensorCore::Tensor<float>& pred, const TensorCore::Tensor<float>& target) -> float {
        size_t correct = 0;
        size_t size = pred.NumElements();

        for (size_t i = 0; i < size; ++i) {
            int predict = (pred[i] > 0) ? 1 : 0;

            if (predict == static_cast<int>(target[i])) {
                correct++;
            }
        }

        return static_cast<float>(correct) / size;
    });

    trainer.OnEpochEnd =
        [](const EpochStats<float>& stats) {

        if (stats.epoch % 500 == 0) {
            std::cout
                << "Epoch " << stats.epoch
                << " | Train Loss: " << stats.trainLoss
                << " | Train Accuracy: "
                << (stats.trainMetrics.at("Accuracy") * 100) << "%"
                << " | Val Loss: "
                << stats.valLoss
                << " | Val Accuracy: "
                << (stats.valMetrics.at("Accuracy") * 100) << "%"
                << '\n';

            std::cout << "Learning Rates: ";

            for (auto& lr : stats.learningRates) {
                std::cout << lr << " ";
            }

            std::cout << std::endl;
        }
    };

    trainer.Fit(x, y, x, y, 10000, 4);

    for (auto& [name, p] : namedParams) {
        std::cout << "Name: " << name << std::endl;
        std::cout << "Values: " << std::endl;

        auto& tensor = p.get().Data();
        auto size = tensor.NumElements();

        for (size_t i = 0; i < size; ++i) {
            std::cout << tensor[i] << " ";

            if (i == size - 1) {
                std::cout << std::endl;
            }
        }
    }
}

int main() {
    ArenaAllocator allocator;

    // Model Training Test
    TestXOR(allocator);

    // Model Saving Test
    Sequential<float> modelA;
    modelA.EmplaceNamed<LinearLayer<float>>("layer1", 2, 8, allocator, InitType::HeUniform);
    modelA.EmplaceNamed<LeakyReLULayer<float>>("leakyReLU");
    modelA.EmplaceNamed<LinearLayer<float>>("layer2", 8, 1, allocator, InitType::HeUniform);

    Checkpoint::Save(modelA, "../../models/xorV2.ckpt");

    Sequential<float> modelB;
    modelB.EmplaceNamed<LinearLayer<float>>("layer1", 2, 8, allocator, InitType::HeUniform);
    modelB.EmplaceNamed<LeakyReLULayer<float>>("leakyReLU");
    modelB.EmplaceNamed<LinearLayer<float>>("layer2", 8, 1, allocator, InitType::HeUniform);

    Checkpoint::Load(modelB, "../../models/xorV2.ckpt");

    auto paramsA = modelA.GetParameters();
    auto paramsB = modelB.GetParameters();

    bool equal = true;

    for (size_t p = 0; p < paramsA.size(); ++p) {
        auto& tensorA = paramsA[p].get().Data();
        auto& tensorB = paramsB[p].get().Data();

        if (tensorA.NumElements() != tensorB.NumElements()) {
            equal = false;
            break;
        }

        for (size_t i = 0; i < tensorA.NumElements(); ++i) {
            if (tensorA[i] != tensorB[i]) {
                equal = false;
                break;
            }
        }
    }

    std::cout << (equal ? "Checkpoint PASSED" : "Checkpoint FAILED") << '\n';

    return 0;
}