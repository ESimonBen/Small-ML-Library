/// trainerTests.cpp
#include <doctest/doctest.h>
#include <mlCore/training/trainer.h>
#include <mlCore/module/sequential.h>
#include <mlCore/module/layers/layers.h>
#include <mlCore/optimizers/optimizers.h>
#include <mlCore/schedulers/schedulers.h>
#include <mlCore/operations/operations.h>

using namespace MLCore::NN;
using namespace MLCore::Utils;
using namespace MLCore::Training;
using namespace MLCore::TensorCore;
using namespace MLCore::Optimizers;
using namespace MLCore::Schedulers;
using namespace MLCore::Operations;

TEST_SUITE("Trainer Tests") {
	TEST_CASE("Trainer Constructor") {
		Sequential<float> model;
		model.Emplace<LinearLayer<float>>(2, 3);
		model.Emplace<ReLULayer<float>>();

		auto parameters = model.GetParameters();

		SGD opt{ parameters, 0.1f };

		Trainer<float> trainer{ model, opt,
			[&](const auto& pred, const auto& target) {
				return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean);
			}
		};

		CHECK(&trainer.GetOptimizer() == &opt);
		CHECK_FALSE(trainer.HasScheduler());
		CHECK(trainer.GetScheduler() == nullptr);
	}

	TEST_CASE("Trainer Scheduler") {
		SUBCASE("Trainer can add a scheduler") {
			Sequential<float> model;
			model.Emplace<LinearLayer<float>>(2, 3);
			model.Emplace<ReLULayer<float>>();

			auto parameters = model.GetParameters();

			SGD opt{ parameters, 0.1f };

			Trainer<float> trainer{ model, opt,
				[&](const auto& pred, const auto& target) {
					return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean);
				}
			};

			StepLR scheduler{ opt, 2, 0.5f };

			trainer.SetScheduler(scheduler, SchedulerStepMode::Epoch);

			CHECK(trainer.HasScheduler());
			CHECK(trainer.GetScheduler() == &scheduler);
		}

		SUBCASE("Scheduler epoch mode correctly updates learning rates") {
			Sequential<float> model;
			model.Emplace<LinearLayer<float>>(2, 3);
			model.Emplace<ReLULayer<float>>();
			model.Emplace<LinearLayer<float>>(3, 1);

			auto parameters = model.GetParameters();

			SGD opt{ parameters, 0.1f };

			Trainer<float> trainer{ model, opt,
				[&](const auto& pred, const auto& target) {
					return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean);
				}
			};

			StepLR scheduler{ opt, 3, 0.5f };

			trainer.SetScheduler(scheduler, SchedulerStepMode::Epoch);

			auto inputs = Tensor<float>::Ones({ 3, 2 });
			auto targets = Tensor<float>::Custom({ 3, 1 }, 3.0f);

			trainer.Fit(inputs, targets, 3, 3);

			auto& groups = opt.ParamGroups();

			CHECK(groups.size() == 1);

			auto learningRate = groups.at(0).learningRate;

			CHECK(learningRate == 0.05f);
		}

		SUBCASE("Scheduler batch mode correctly updates learning rates") {
			Sequential<float> model;
			model.Emplace<LinearLayer<float>>(2, 3);
			model.Emplace<ReLULayer<float>>();
			model.Emplace<LinearLayer<float>>(3, 1);

			auto parameters = model.GetParameters();

			SGD opt{ parameters, 0.1f };

			Trainer<float> trainer{ model, opt,
				[&](const auto& pred, const auto& target) {
					return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean);
				}
			};

			StepLR scheduler{ opt, 3, 0.5f };

			trainer.SetScheduler(scheduler, SchedulerStepMode::Batch);

			auto inputs = Tensor<float>::Ones({ 3, 2 });
			auto targets = Tensor<float>::Custom({ 3, 1 }, 3.0f);

			trainer.Fit(inputs, targets, 3, 3);

			auto& groups = opt.ParamGroups();

			CHECK(groups.size() == 1);

			auto learningRate = groups.at(0).learningRate;

			CHECK(learningRate == 0.05f);
		}
	}

	TEST_CASE("Trainer Metric Functions") {
		SUBCASE("AddMetric stores metrics") {
			Sequential<float> model;
			model.Emplace<LinearLayer<float>>(2, 3);
			model.Emplace<ReLULayer<float>>();
			model.Emplace<LinearLayer<float>>(3, 1);

			auto parameters = model.GetParameters();

			SGD opt{ parameters, 0.1f };

			Trainer<float> trainer{ model, opt,
				[&](const auto& pred, const auto& target) {
					return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean);
				}
			};

			auto inputs = Tensor<float>::Ones({ 3, 2 });
			auto targets = Tensor<float>::Custom({ 3, 1 }, 3.0f);

			auto metricCalled = false;

			auto metric = [&](const Tensor<float>& pred, const Tensor<float>& target) -> float {
				size_t correct = 0;
				size_t size = pred.NumElements();

				for (size_t i = 0; i < size; ++i) {
					int predict = (pred[i] > 0) ? 1 : 0;

					if (predict == static_cast<int>(target[i])) {
						correct++;
					}
				}

				metricCalled = true;

				return static_cast<float>(correct) / size;
			};

			trainer.AddMetric("Accuracy", metric);

			trainer.Fit(inputs, targets, 1, 3);
			
			CHECK(metricCalled);
		}

		SUBCASE("Metric function is called once per batch") {
			Sequential<float> model;
			model.Emplace<LinearLayer<float>>(2, 3);
			model.Emplace<ReLULayer<float>>();
			model.Emplace<LinearLayer<float>>(3, 1);

			auto parameters = model.GetParameters();

			SGD opt{ parameters, 0.1f };

			Trainer<float> trainer{ model, opt,
				[&](const auto& pred, const auto& target) {
					return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean);
				}
			};

			auto inputs = Tensor<float>::Ones({ 3, 2 });
			auto targets = Tensor<float>::Custom({ 3, 1 }, 3.0f);

			int metricCalls = 0;

			auto metric = [&](const Tensor<float>& pred, const Tensor<float>& target) -> float {
				size_t correct = 0;
				size_t size = pred.NumElements();

				for (size_t i = 0; i < size; ++i) {
					int predict = (pred[i] > 0) ? 1 : 0;

					if (predict == static_cast<int>(target[i])) {
						correct++;
					}
				}

				metricCalls++;

				return static_cast<float>(correct) / size;
				};

			trainer.AddMetric("Accuracy", metric);

			trainer.Fit(inputs, targets, 1, 1);

			CHECK(metricCalls == 3);
		}

		SUBCASE("Multiple metric functions get called simultaneously") {
			Sequential<float> model;
			model.Emplace<LinearLayer<float>>(2, 3);
			model.Emplace<ReLULayer<float>>();
			model.Emplace<LinearLayer<float>>(3, 1);

			auto parameters = model.GetParameters();

			SGD opt{ parameters, 0.1f };

			Trainer<float> trainer{ model, opt,
				[&](const auto& pred, const auto& target) {
					return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean);
				}
			};

			auto inputs = Tensor<float>::Ones({ 3, 2 });
			auto targets = Tensor<float>::Custom({ 3, 1 }, 3.0f);

			int metricCallsA = 0;
			int metricCallsB = 0;

			auto metricA = [&](const Tensor<float>& pred, const Tensor<float>& target) -> float {
				size_t correct = 0;
				size_t size = pred.NumElements();

				for (size_t i = 0; i < size; ++i) {
					int predict = (pred[i] > 0) ? 1 : 0;

					if (predict == static_cast<int>(target[i])) {
						correct++;
					}
				}

				metricCallsA++;

				return static_cast<float>(correct) / size;
			};

			auto metricB = [&](const Tensor<float>& pred, const Tensor<float>& target) -> float {
				metricCallsB++;
				return MeanAll(pred)[0] / MeanAll(target)[0];
			};

			trainer.AddMetric("Accuracy", metricA);
			trainer.AddMetric("Random", metricB);

			trainer.Fit(inputs, targets, 1, 1);

			CHECK(metricCallsA == 3);
			CHECK(metricCallsB == 3);
		}
	}

	TEST_CASE("Trainer State") {
		SUBCASE("Default Trainer State") {
			Sequential<float> model;
			model.Emplace<LinearLayer<float>>(2, 3);
			model.Emplace<ReLULayer<float>>();
			model.Emplace<LinearLayer<float>>(3, 1);

			auto parameters = model.GetParameters();

			SGD opt{ parameters, 0.1f };

			Trainer<float> trainer{ model, opt,
				[&](const auto& pred, const auto& target) {
					return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean);
				}
			};

			auto inputs = Tensor<float>::Ones({ 3, 2 });
			auto targets = Tensor<float>::Custom({ 3, 1 }, 3.0f);

			auto state = trainer.GetState();

			CHECK(state.currentEpoch == 0);
			CHECK(state.globalStep == 0);
			CHECK_FALSE(state.hasBestMetric);
		}

		SUBCASE("Trainer state changes after training") {
			Sequential<float> model;
			model.Emplace<LinearLayer<float>>(2, 3);
			model.Emplace<ReLULayer<float>>();
			model.Emplace<LinearLayer<float>>(3, 1);

			auto parameters = model.GetParameters();

			SGD opt{ parameters, 0.1f };

			auto lossFn = [&](const auto& pred, const auto& target) {
				return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean);
			};

			Trainer<float> trainer{ model, opt, lossFn};

			auto trainInputs = Tensor<float>::Ones({ 3, 2 });
			auto trainTargets = Tensor<float>::Custom({ 3, 1 }, 3.0f);
			auto valInputs = Tensor<float>::Zeros({ 3, 2 });
			auto valTargets = Tensor<float>::Custom({ 3, 1 }, 2.0f);

			auto metric = [&](const Tensor<float>& pred, const Tensor<float>& target) -> float {
				size_t correct = 0;
				size_t size = pred.NumElements();

				for (size_t i = 0; i < size; ++i) {
					int predict = (pred[i] > 0) ? 1 : 0;

					if (predict == static_cast<int>(target[i])) {
						correct++;
					}
				}

				return static_cast<float>(correct) / size;
			};

			trainer.AddMetric("Accuracy", metric);

			trainer.OnEpochEnd = [](const EpochStats<float>& stats) {
				return;
			};

			trainer.Fit(trainInputs, trainTargets, valInputs, valTargets, 1, 1);

			auto state = trainer.GetState();

			CHECK(state.currentEpoch == 1);
			CHECK(state.globalStep == 3);
			CHECK(state.hasBestMetric);
		}

		SUBCASE("LoadState restores training state") {
			Sequential<float> model;
			model.Emplace<LinearLayer<float>>(2, 3);
			model.Emplace<ReLULayer<float>>();
			model.Emplace<LinearLayer<float>>(3, 1);

			auto parameters = model.GetParameters();

			SGD opt{ parameters, 0.1f };

			auto lossFn = [&](const auto& pred, const auto& target) {
				return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean);
			};

			Trainer<float> trainer{ model, opt, lossFn};

			auto trainInputs = Tensor<float>::Ones({ 3, 2 });
			auto trainTargets = Tensor<float>::Custom({ 3, 1 }, 3.0f);
			auto valInputs = Tensor<float>::Zeros({ 3, 2 });
			auto valTargets = Tensor<float>::Custom({ 3, 1 }, 2.0f);

			auto metric = [&](const Tensor<float>& pred, const Tensor<float>& target) -> float {
				size_t correct = 0;
				size_t size = pred.NumElements();

				for (size_t i = 0; i < size; ++i) {
					int predict = (pred[i] > 0) ? 1 : 0;

					if (predict == static_cast<int>(target[i])) {
						correct++;
					}
				}

				return static_cast<float>(correct) / size;
				};

			trainer.AddMetric("Accuracy", metric);

			trainer.OnEpochEnd = [](const EpochStats<float>& stats) {
				return;
				};

			trainer.Fit(trainInputs, trainTargets, valInputs, valTargets, 1, 1);

			auto state = trainer.GetState();

			CHECK(state.currentEpoch == 1);
			CHECK(state.globalStep == 3);
			CHECK(state.hasBestMetric);

			Trainer<float> trainer2{ model, opt, lossFn };
			auto state2 = trainer2.GetState();

			CHECK(state2.currentEpoch == 0);
			CHECK(state2.globalStep == 0);
			CHECK_FALSE(state2.hasBestMetric);

			trainer2.LoadState(state);

			auto state3 = trainer2.GetState();

			CHECK(state3.currentEpoch == 1);
			CHECK(state3.globalStep == 3);
			CHECK(state3.hasBestMetric);
		}
	}
}
