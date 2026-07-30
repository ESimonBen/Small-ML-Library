/// datasetTests.cpp
#include <doctest/doctest.h>
#include <mlCore/data/dataLoader.h>
#include <mlCore/data/tensorDataset.h>

using namespace MLCore::Data;
using namespace MLCore::Utils;
using namespace MLCore::TensorCore;

TEST_SUITE("Data Tests") {
	TEST_CASE("Tensor Dataset Tests") {
		SUBCASE("Tensor dataset constructor") {
			auto input = Tensor<float>::Custom({ 2, 3 }, 3.0f);
			auto target = Tensor<float>::Ones({ 2, 3 });

			TensorDataset dataset{ input, target };

			CHECK(dataset.Size() == 2);
		}

		SUBCASE("GetItem correctly splits input and target batches") {
			auto inputs = Tensor<float>::Custom({ 2, 3 }, 3.0f);
			auto targets = Tensor<float>::Ones({ 2, 1 });

			TensorDataset dataset{ inputs, targets };

			auto item = dataset.GetItem(0);

			auto input = item.first;
			auto target = item.second;


			CHECK(input.GetShape() == Shape(1, 3));
			for (auto& val : input) {
				CHECK(val == 3.0f);
			}

			CHECK(target.GetShape() == Shape(1, 1));
			for (auto& val : target) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("GetItem throws on out of bounds index") {
			auto inputs = Tensor<float>::Custom({ 2, 3 }, 4.0f);
			auto targets = Tensor<float>::Zeros({ 2, 1 });

			TensorDataset dataset{ inputs, targets };

			CHECK_THROWS_AS(dataset.GetItem(2.0f), std::out_of_range);
		}
	}

	TEST_CASE("Data Loader Tests") {
		SUBCASE("Data Loader Constructor") {
			auto inputs = Tensor<float>::Custom({ 2, 3 }, 6.0f);
			auto targets = Tensor<float>::Ones({ 2, 1 });

			TensorDataset dataset{ inputs, targets };
			DataLoader loader{ dataset, 1 };

			CHECK(loader.HasNext());
		}

		SUBCASE("Each iteration in the Data Loader splits by batch size") {
			auto inputs = Tensor<float>::Custom({ 10, 3 }, 4.0f);
			auto targets = Tensor<float>::Ones({ 10, 1 });

			TensorDataset dataset{ inputs, targets };
			DataLoader loader{ dataset, 4 };

			auto iter1 = loader.Next();

			CHECK(iter1.first.GetShape() == Shape(4, 3));
			CHECK(iter1.second.GetShape() == Shape(4, 1));

			auto iter2 = loader.Next();

			CHECK(iter2.first.GetShape() == Shape(4, 3));
			CHECK(iter2.second.GetShape() == Shape(4, 1));

			auto iter3 = loader.Next();

			CHECK(iter3.first.GetShape() == Shape(2, 3));
			CHECK(iter3.second.GetShape() == Shape(2, 1));

			CHECK_THROWS_AS(loader.Next(), std::out_of_range);
		}
	}
}