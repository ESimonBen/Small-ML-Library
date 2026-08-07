/// serializationTests.cpp
#include <set>
#include <doctest/doctest.h>
#include <mlCore/training/trainer.h>
#include <mlCore/module/sequential.h>
#include <mlCore/module/layers/layers.h>
#include <mlCore/schedulers/schedulers.h>
#include <mlCore/optimizers/optimizers.h>
#include <mlCore/operations/operations.h>
#include <mlCore/serialization/checkpoint.h>

using namespace MLCore::NN;
using namespace MLCore::Training;
using namespace MLCore::Optimizers;
using namespace MLCore::TensorCore;
using namespace MLCore::Schedulers;
using namespace MLCore::Operations;
using namespace MLCore::Serialization;

TEST_SUITE("Serialization Tests") {
	TEST_CASE("Binary Archive Tests") {
		SUBCASE("Write and Read basic data to a file") {
			{
				std::ofstream out{ "models/testFiles/test1.ckpt", std::ios::binary };

				BinaryWriter writer(out);

				int test = 4;
				writer.Write(test);
			}

			std::ifstream in{ "models/testFiles/test1.ckpt", std::ios::binary };

			BinaryReader reader{ in };

			int number;
			reader.Read(number);

			CHECK(number == 4);
		}

		SUBCASE("Write and Read an array/string to a file") {
			{
				std::ofstream out{ "models/testFiles/test2.ckpt", std::ios::binary };

				BinaryWriter writer{ out };

				std::string string = "Hello!";
				size_t size = string.size();

				writer.Write(size);
				writer.WriteArray(string.data(), size);
			}

			std::ifstream in{ "models/testFiles/test2.ckpt", std::ios::binary };

			BinaryReader reader{ in };

			size_t size;
			reader.Read(size);

			CHECK(size == 6);

			std::string string(size, '\0');
			reader.ReadArray(string.data(), size);

			CHECK(string == "Hello!");
		}

		SUBCASE("Write and Read a tensor to a file") {
			{
				std::ofstream out{ "models/testFiles/test3.ckpt", std::ios::binary };

				BinaryWriter writer{ out };

				auto tensor = Tensor<float>::Custom({ 3, 2 }, 5.0f);

				writer.WriteTensor(tensor);
			}

			std::ifstream in{ "models/testFiles/test3.ckpt", std::ios::binary };

			BinaryReader reader{ in };

			Tensor<float> tensor{{3, 2}};

			reader.ReadTensor(tensor);

			for (auto& val : tensor) {
				CHECK(val == 5.0f);
			}
		}
	}

	TEST_CASE("Checkpoint Tests") {
		SUBCASE("Save and Load model parameters") {
			Sequential<float> model;
			model.Emplace<LinearLayer<float>>(2, 3, InitType::HeUniform);
			model.Emplace<LeakyReLULayer<float>>(0.1f);
			model.Emplace<LinearLayer<float>>(3, 1, InitType::HeUniform);

			std::set<float> seenVals;
			auto parameters = model.GetParameters();
			
			for (auto& ref : parameters) {
				auto& param = ref.get();
				auto& tensor = param.Data();

				for (auto val : tensor) {
					seenVals.insert(val);
				}
			}

			Checkpoint::Save(model, "models/testFiles/checkpointTest1.ckpt");

			Sequential<float> model2;
			model2.Emplace<LinearLayer<float>>(2, 3);
			model2.Emplace<LeakyReLULayer<float>>(0.1f);
			model2.Emplace<LinearLayer<float>>(3, 1);

			Checkpoint::Load(model2, "models/testFiles/checkpointTest1.ckpt");

			auto parameters2 = model2.GetParameters();

			for (auto& ref : parameters2) {
				auto& param = ref.get();
				auto& tensor = param.Data();

				for (auto val : tensor) {
					CHECK(seenVals.contains(val));
				}
			}
		}

		SUBCASE("Save and Load named parameters") {
			Sequential<float> model;
			model.EmplaceNamed<LinearLayer<float>>("fc1", 2, 3, InitType::HeUniform);
			model.EmplaceNamed<LeakyReLULayer<float>>("leakyRelu", 0.1f);
			model.EmplaceNamed<LinearLayer<float>>("fc2" ,3, 1, InitType::HeUniform);

			std::set<float> seenVals;
			auto parameters = model.GetParameters();

			for (auto& ref : parameters) {
				auto& param = ref.get();
				auto& tensor = param.Data();

				for (auto val : tensor) {
					seenVals.insert(val);
				}
			}

			Checkpoint::Save(model, "models/testFiles/checkpointTest2.ckpt");

			Sequential<float> model2;
			model2.EmplaceNamed<LinearLayer<float>>("fc1", 2, 3);
			model2.EmplaceNamed<LeakyReLULayer<float>>("leakyRelu", 0.1f);
			model2.EmplaceNamed<LinearLayer<float>>("fc2", 3, 1);

			Checkpoint::Load(model2, "models/testFiles/checkpointTest2.ckpt");

			auto parameters2 = model2.GetParameters();

			for (auto& ref : parameters2) {
				auto& param = ref.get();
				auto& tensor = param.Data();

				for (auto val : tensor) {
					CHECK(seenVals.contains(val));
				}
			}
		}

		SUBCASE("Save and Load optimizer") {
			Sequential<float> model;
			model.Emplace<LinearLayer<float>>(2, 3, InitType::HeUniform);
			model.Emplace<LeakyReLULayer<float>>(0.1f);
			model.Emplace<LinearLayer<float>>(3, 1, InitType::HeUniform);

			std::set<float> seenVals;
			auto parameters = model.GetParameters();

			for (auto& ref : parameters) {
				auto& param = ref.get();
				auto& tensor = param.Data();

				for (auto val : tensor) {
					seenVals.insert(val);
				}
			}

			SGD opt{ parameters, 0.1f, 0.5f };

			Checkpoint::Save(model, "models/testFiles/checkpointTest3.ckpt", &opt);

			Sequential<float> model2;
			model2.Emplace<LinearLayer<float>>(2, 3);
			model2.Emplace<LeakyReLULayer<float>>(0.1f);
			model2.Emplace<LinearLayer<float>>(3, 1);

			auto parameters2 = model2.GetParameters();

			SGD opt2{ parameters2, 0.2f, 0.2f }; /// Learning rate and weight decay should be different after loading

			Checkpoint::Load(model2, "models/testFiles/checkpointTest3.ckpt", &opt2);

			for (auto& ref : parameters2) {
				auto& param = ref.get();
				auto& tensor = param.Data();

				for (auto val : tensor) {
					CHECK(seenVals.contains(val));
				}
			}

			auto& groups = opt2.ParamGroups();

			CHECK(groups.size() == 1);

			for (auto& group : groups) {
				CHECK(group.learningRate == 0.1f);
				CHECK(group.weightDecay == 0.5f);
			}
		}

		SUBCASE("Save and Load learning rate scheduler") {
			Sequential<float> model;
			model.Emplace<LinearLayer<float>>(2, 3, InitType::HeUniform);
			model.Emplace<LeakyReLULayer<float>>(0.1f);
			model.Emplace<LinearLayer<float>>(3, 1, InitType::HeUniform);

			std::set<float> seenVals;
			auto parameters = model.GetParameters();

			for (auto& ref : parameters) {
				auto& param = ref.get();
				auto& tensor = param.Data();

				for (auto val : tensor) {
					seenVals.insert(val);
				}
			}

			SGD opt{ parameters, 0.1f, 0.5f };

			StepLR scheduler{ opt, 2, 0.9f };

			Checkpoint::Save(model, "models/testFiles/checkpointTest4.ckpt", &opt, &scheduler);

			Sequential<float> model2;
			model2.Emplace<LinearLayer<float>>(2, 3);
			model2.Emplace<LeakyReLULayer<float>>(0.1f);
			model2.Emplace<LinearLayer<float>>(3, 1);

			auto parameters2 = model2.GetParameters();

			SGD opt2{ parameters2, 0.2f, 0.2f }; /// Learning rate and weight decay should be different after loading

			StepLR scheduler2{ opt2, 3, 0.2f }; /// Step size and gamma should be different after loading

			Checkpoint::Load(model2, "models/testFiles/checkpointTest4.ckpt", &opt2, &scheduler2);

			for (auto& ref : parameters2) {
				auto& param = ref.get();
				auto& tensor = param.Data();

				for (auto val : tensor) {
					CHECK(seenVals.contains(val));
				}
			}

			auto& groups = opt2.ParamGroups();

			CHECK(groups.size() == 1);

			for (auto& group : groups) {
				CHECK(group.learningRate == 0.1f);
				CHECK(group.weightDecay == 0.5f);
			}

			CHECK(scheduler2.StepSize() == 2);
			CHECK(scheduler2.Gamma() == 0.9f);
		}

		SUBCASE("Save and Load training state") {
			Sequential<float> model;
			model.Emplace<LinearLayer<float>>(2, 3, InitType::HeUniform);
			model.Emplace<LeakyReLULayer<float>>(0.1f);
			model.Emplace<LinearLayer<float>>(3, 1, InitType::HeUniform);

			std::set<float> seenVals;
			auto parameters = model.GetParameters();

			for (auto& ref : parameters) {
				auto& param = ref.get();
				auto& tensor = param.Data();

				for (auto val : tensor) {
					seenVals.insert(val);
				}
			}

			SGD opt{ parameters, 0.1f, 0.5f };

			auto lossFn = [](const Tensor<float>& pred, const Tensor<float>& target) {
				return BinaryCrossEntropyWithLogits(pred, target, Reduction::Mean);
			};

			Trainer<float> trainer{ model, opt, lossFn };

			auto inputs = Tensor<float>::Ones({ 3, 2 });
			auto targets = Tensor<float>::Custom({ 3, 1 }, 5.0f);

			trainer.Fit(inputs, targets, 5, 2);

			auto state = trainer.GetState();

			Checkpoint::Save<float>(model, "models/testFiles/checkpointTest5.ckpt", nullptr, nullptr, &state);

			Trainer<float> trainer2{ model, opt, lossFn };

			auto state2 = trainer2.GetState();

			Checkpoint::Load<float>(model, "models/testFiles/checkpointTest5.ckpt", nullptr, nullptr, &state2);

			trainer2.LoadState(state2);

			CHECK(state2.currentEpoch == 5);
			CHECK(state2.globalStep == 10);
			CHECK(state2.hasBestMetric == false);
		}
	}
}