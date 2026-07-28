/// schedulerTests.cpp
#include <doctest/doctest.h>
#include <mlCore/optimizers/sgd.h>
#include <mlCore/schedulers/schedulers.h>

using namespace MLCore::NN;
using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::TensorCore;
using namespace MLCore::Optimizers;
using namespace MLCore::Schedulers;

TEST_SUITE("Learning Rate Scheduler Tests") {
	TEST_CASE("Step Scheduler Tests") {
		SUBCASE("Step Scheduler Constructor") {
			auto weight = Tensor<float>::Custom({ 2, 3 }, 5.0f);

			Parameter param{ weight };
			std::vector<std::reference_wrapper<Parameter<float>>> params{ param };

			SGD opt{ params, 0.1f };
			StepLR scheduler{ opt, 2, 0.5f };

			auto learningRates = scheduler.GetLastLRs();

			CHECK(learningRates.size() == 1);
			CHECK(learningRates.at(0) == 0.1f);
		}

		SUBCASE("Step Scheduler correctly updates learning rates") {
			auto weight = Tensor<float>::Custom({ 2, 3 }, 5.0f);

			Parameter param{ weight };
			std::vector<std::reference_wrapper<Parameter<float>>> params{ param };

			SGD opt{ params, 0.1f };
			StepLR scheduler{ opt, 2, 0.5f };

			scheduler.UpdateLR();

			auto& paramGroups = opt.ParamGroups();

			CHECK(paramGroups.at(0).learningRate == 0.1f);

			scheduler.UpdateLR();

			CHECK(paramGroups.at(0).learningRate == 0.05f);
		}

		SUBCASE("Step Scheduler correctly updates multiple learning rates") {
			auto weight1 = Tensor<float>::Custom({ 2, 3 }, 3.0f);
			auto weight2 = Tensor<float>::Custom({ 2, 3 }, 5.0f);
			auto weight3 = Tensor<float>::Custom({ 2, 3 }, 7.0f);

			Parameter param1{ weight1 };
			Parameter param2{ weight2 };
			Parameter param3{ weight3 };

			ParameterGroup group1{ {param1}, 0.1f };
			ParameterGroup group2{ {param2}, 0.2f };
			ParameterGroup group3{ {param3}, 0.3f };

			std::vector<ParameterGroup<float>> groups{ group1, group2, group3 };

			SGD opt{ groups };
			StepLR scheduler{ opt, 2, 0.5f };

			scheduler.UpdateLR();

			auto& paramGroups = opt.ParamGroups();

			CHECK(paramGroups.at(0).learningRate == 0.1f);
			CHECK(paramGroups.at(1).learningRate == 0.2f);
			CHECK(paramGroups.at(2).learningRate == 0.3f);

			scheduler.UpdateLR();

			CHECK(paramGroups.at(0).learningRate == 0.05f);
			CHECK(paramGroups.at(1).learningRate == 0.1f);
			CHECK(paramGroups.at(2).learningRate == 0.15f);
		}
	}

	TEST_CASE("Exponential Scheduler Tests") {
		SUBCASE("Exponential Scheduler Constructor") {
			auto weight = Tensor<float>::Custom({ 2, 3 }, 5.0f);

			Parameter param{ weight };
			std::vector<std::reference_wrapper<Parameter<float>>> params{ param };

			SGD opt{ params, 0.1f };
			ExponentialLR scheduler{ opt, 0.5f };

			auto learningRates = scheduler.GetLastLRs();

			CHECK(learningRates.size() == 1);
			CHECK(learningRates.at(0) == 0.1f);
		}

		SUBCASE("Exponential Scheduler correctly updates learning rates") {
			auto weight = Tensor<float>::Custom({ 2, 3 }, 5.0f);

			Parameter param{ weight };
			std::vector<std::reference_wrapper<Parameter<float>>> params{ param };

			SGD opt{ params, 0.1f };
			ExponentialLR scheduler{ opt, 0.5f };

			scheduler.UpdateLR();

			auto& paramGroups = opt.ParamGroups();

			CHECK(paramGroups.at(0).learningRate == 0.05f);

			scheduler.UpdateLR();

			CHECK(paramGroups.at(0).learningRate == 0.025f);
		}

		SUBCASE("Exponential Scheduler correctly updates multiple learning rates") {
			auto weight1 = Tensor<float>::Custom({ 2, 3 }, 3.0f);
			auto weight2 = Tensor<float>::Custom({ 2, 3 }, 5.0f);
			auto weight3 = Tensor<float>::Custom({ 2, 3 }, 7.0f);

			Parameter param1{ weight1 };
			Parameter param2{ weight2 };
			Parameter param3{ weight3 };

			ParameterGroup group1{ {param1}, 0.1f };
			ParameterGroup group2{ {param2}, 0.2f };
			ParameterGroup group3{ {param3}, 0.3f };

			std::vector<ParameterGroup<float>> groups{ group1, group2, group3 };

			SGD opt{ groups };
			ExponentialLR scheduler{ opt, 0.5f };

			scheduler.UpdateLR();

			auto& paramGroups = opt.ParamGroups();

			CHECK(paramGroups.at(0).learningRate == 0.05f);
			CHECK(paramGroups.at(1).learningRate == 0.1f);
			CHECK(paramGroups.at(2).learningRate == 0.15f);

			scheduler.UpdateLR();

			CHECK(paramGroups.at(0).learningRate == 0.025f);
			CHECK(paramGroups.at(1).learningRate == 0.05f);
			CHECK(paramGroups.at(2).learningRate == 0.075f);
		}
	}
}