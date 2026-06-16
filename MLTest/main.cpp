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

void TestXORSave(ArenaAllocator& allocator) {
    std::cout << "=== XOR Nonlinear Test ===\n";

    // -----------------------------
    // 1. Dataset (XOR)
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
    // 2. Base Model
    // -----------------------------
    Sequential<float> model;

    model.EmplaceNamed<LinearLayer<float>>("layer1", 2, 8, allocator, InitType::HeUniform);
    model.EmplaceNamed<LeakyReLULayer<float>>("leakyReLU");
    model.EmplaceNamed<LinearLayer<float>>("layer2", 8, 1, allocator, InitType::HeUniform);

    Checkpoint::Save(model, "../../models/base_model.ckpt");

    // Model A (Test Continuous)
    Sequential<float> modelA;
    modelA.EmplaceNamed<LinearLayer<float>>("layer1", 2, 8, allocator, InitType::HeUniform);
    modelA.EmplaceNamed<LeakyReLULayer<float>>("leakyReLU");
    modelA.EmplaceNamed<LinearLayer<float>>("layer2", 8, 1, allocator, InitType::HeUniform);

    Checkpoint::Load(modelA, "../../models/base_model.ckpt");

    // Collect parameters
    auto paramsA = modelA.GetParameters();
    auto namedParamsA = modelA.GetNamedParameters();

    std::cout << "Params at Beginning (Model A)" << std::endl;
    for (auto& [name, p] : namedParamsA) {
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

    SGDMomentum<float> optA{ paramsA, 0.1f, 0.1f };

    StepLR<float> schedulerA{ optA, 1000, .99f };

    // -----------------------------
    // 3. Training loop
    // -----------------------------
    Trainer<float> trainerA{ modelA, optA,
        [&](const auto& pred, const auto& target) {
            return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean, allocator);
        }
    };

    trainerA.SetScheduler(schedulerA, SchedulerStepMode::Epoch);

    trainerA.AddMetric("Accuracy", [](const TensorCore::Tensor<float>& pred, const TensorCore::Tensor<float>& target) -> float {
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

    trainerA.OnEpochEnd =
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

    trainerA.Fit(x, y, x, y, 10000, 4);

    std::cout << std::endl;
    std::cout << "Params at End (Model A)" << std::endl;
    for (auto& [name, p] : namedParamsA) {
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

    std::cout << std::endl;

    // Model B (Pause/Resume)
    Sequential<float> modelB;
    modelB.EmplaceNamed<LinearLayer<float>>("layer1", 2, 8, allocator, InitType::HeUniform);
    modelB.EmplaceNamed<LeakyReLULayer<float>>("leakyReLU");
    modelB.EmplaceNamed<LinearLayer<float>>("layer2", 8, 1, allocator, InitType::HeUniform);

    Checkpoint::Load(modelB, "../../models/base_model.ckpt");

    // Collect parameters
    auto paramsB = modelB.GetParameters();
    auto namedParamsB = modelB.GetNamedParameters();

    std::cout << "Params at Beginning (Model B)" << std::endl;
    for (auto& [name, p] : namedParamsB) {
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

    SGDMomentum<float> optB{ paramsB, 0.1f, 0.1f };

    StepLR<float> schedulerB{ optB, 1000, .99f };

    // -----------------------------
    // 3. Training loop
    // -----------------------------
    Trainer<float> trainerB{ modelB, optB,
        [&](const auto& pred, const auto& target) {
            return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean, allocator);
        }
    };

    trainerB.SetScheduler(schedulerB, SchedulerStepMode::Epoch);

    trainerB.AddMetric("Accuracy", [](const TensorCore::Tensor<float>& pred, const TensorCore::Tensor<float>& target) -> float {
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

    trainerB.OnEpochEnd =
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

    trainerB.Fit(x, y, x, y, 5000, 4);

    TrainerState<float> stateB = trainerB.GetState();

    Checkpoint::Save(modelB, "../../models/modelB.ckpt", &optB, &schedulerB, &stateB);

    // Resume Model
    Sequential<float> modelC;
    modelC.EmplaceNamed<LinearLayer<float>>("layer1", 2, 8, allocator, InitType::HeUniform);
    modelC.EmplaceNamed<LeakyReLULayer<float>>("leakyReLU");
    modelC.EmplaceNamed<LinearLayer<float>>("layer2", 8, 1, allocator, InitType::HeUniform);

    // Collect parameters
    auto paramsC = modelC.GetParameters();
    auto namedParamsC = modelC.GetNamedParameters();

    SGDMomentum<float> optC{ paramsC, 0.1f, 0.1f };

    StepLR<float> schedulerC{ optC, 1000, .99f };

    // -----------------------------
    // 3. Training loop
    // -----------------------------
    Trainer<float> trainerC{ modelC, optC,
        [&](const auto& pred, const auto& target) {
            return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean, allocator);
        }
    };

    trainerC.SetScheduler(schedulerC, SchedulerStepMode::Epoch);

    trainerC.AddMetric("Accuracy", [](const TensorCore::Tensor<float>& pred, const TensorCore::Tensor<float>& target) -> float {
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

    trainerC.OnEpochEnd =
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

    TrainerState<float> stateC;

    Checkpoint::Load(modelC, "../../models/modelB.ckpt", &optC, &schedulerC, &stateC);
    trainerC.LoadState(stateC);

    trainerC.Fit(x, y, x, y, 5000, 4);

    std::cout << std::endl;
    std::cout << "Params at End (Model B)" << std::endl;
    for (auto& [name, p] : namedParamsC) {
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
    TestXORSave(allocator);

    return 0;
}