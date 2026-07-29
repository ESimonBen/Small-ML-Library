/// layerTests.cpp
#include <doctest/doctest.h>
#include <mlCore/module/sequential.h>
#include <mlCore/module/layers/layers.h>

using namespace MLCore::NN;
using namespace MLCore::Utils;
using namespace MLCore::TensorCore;

TEST_SUITE("Module/Layer Tests") {
	TEST_CASE("Module Tests") {
		SUBCASE("Sequential Module Constructor") {
			Sequential<float> model;

			CHECK(model.IsTraining());
		}

		SUBCASE("Empty Sequential module returns unchanged input") {
			Sequential<float> model;

			auto input = Tensor<float>::Custom({ 2, 3 }, 4.0f);
			auto result = model(input);
			
			size_t size = input.NumElements();

			for (size_t i = 0; i < size; ++i) {
				CHECK(input[i] == result[i]);
			}
		}

		SUBCASE("Multi-layer module calculates forward correctly") {
			Sequential<float> model;
			model.Emplace<LinearLayer<float>>(2, 3);

			auto input = Tensor<float>::Custom({ 3, 2 }, 3.0f);
			auto result = model.Forward(input);

			CHECK(result.GetShape() == Shape(3, 3));
		}

		SUBCASE("Sequential module can add layers") {
			Sequential<float> model;

			model.Emplace<LinearLayer<float>>(2, 4, InitType::Zero);

			CHECK(model.GetParameters().size() == 2); /// For both the weight and bias parameters

			model.Emplace<ReLULayer<float>>();

			CHECK(model.GetParameters().size() == 2); /// ReLU layer has no parameters
		}

		SUBCASE("Sequentual module automatically names parameters") {
			Sequential<float> model;

			model.Emplace<LinearLayer<float>>(2, 4, InitType::Zero);
			model.Emplace<LinearLayer<float>>(4, 3, InitType::Zero);

			auto namedParams = model.GetNamedParameters();

			CHECK(namedParams[0].first == "layer0.weight");
			CHECK(namedParams[1].first == "layer0.bias");
			CHECK(namedParams[2].first == "layer1.weight");
			CHECK(namedParams[3].first == "layer1.bias");
		}

		SUBCASE("Sequential module takes in layer names") {
			Sequential<float> model;

			model.EmplaceNamed<LinearLayer<float>>("fc1", 2, 4, InitType::Zero);
			model.EmplaceNamed<LinearLayer<float>>("fc2", 4, 3, InitType::Zero);

			auto namedParams = model.GetNamedParameters();

			CHECK(namedParams[0].first == "fc1.weight");
			CHECK(namedParams[1].first == "fc1.bias");
			CHECK(namedParams[2].first == "fc2.weight");
			CHECK(namedParams[3].first == "fc2.bias");
		}

		SUBCASE("Module switches between train and evaulate modes") {
			Sequential<float> model;

			model.Train();

			CHECK(model.IsTraining());

			model.Evaluate();

			CHECK_FALSE(model.IsTraining());
		}
	}
}