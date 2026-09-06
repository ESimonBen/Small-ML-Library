/// poolingTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/pooling/pooling.h>

using namespace MLCore::Utils;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Pooling Operation Tests") {
	TEST_CASE("MaxPool1D") {
		SUBCASE("MaxPool1D Calculation") {
			Tensor<float> input{ {1, 1, 6} };
			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool1D(input, 2);

			CHECK(output.GetShape() == Shape(1, 1, 3));

			CHECK(output[0] == 2);
			CHECK(output[1] == 4);
			CHECK(output[2] == 6);
		}

		SUBCASE("MaxPool1D calculates correctly with multiple input and output channels") {
			Tensor<float> input{ {1, 2, 6} };
			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool1D(input, 2);

			CHECK(output.GetShape() == Shape(1, 2, 3));

			CHECK(output[0] == 2);
			CHECK(output[1] == 4);
			CHECK(output[2] == 6);
			CHECK(output[3] == 8);
			CHECK(output[4] == 10);
			CHECK(output[5] == 12);
		}

		SUBCASE("MaxPool1D with ceilMode gives correct shape") {
			Tensor<float> input{ {1, 1, 7} };
			input.Fill(1.0f);

			auto output = MaxPool1D(input, 2, 0, 0, 1, true);

			CHECK(output.GetShape() == Shape(1, 1, 4));

			for (auto& val : output) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("MaxPool1D strides change output dimensions") {
			auto input = Tensor<float>::Ones({ 1, 1, 6 });

			auto output = MaxPool1D(input, 2, 1);

			CHECK(output.GetShape() == Shape(1, 1, 5));

			for (auto& val : output) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("MaxPool1D padding correctly maintains dimensions") {
			auto input = Tensor<float>::Ones({ 1, 1, 6 });

			auto output = MaxPool1D(input, 2, 0, 1);

			CHECK(output.GetShape() == Shape(1, 1, 4));

			for (auto& val : output) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("MaxPool1D dilation correctly maintains dimensions") {
			auto input = Tensor<float>::Ones({ 1, 1, 6 });

			auto output = MaxPool1D(input, 2, 1, 0, 2);

			CHECK(output.GetShape() == Shape(1, 1, 4));

			for (auto& val : output) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("MaxPool1D rejects invalid input dimensions") {
			auto input = Tensor<float>::Ones({ 1, 2 });

			CHECK_THROWS_AS(MaxPool1D(input, 1), std::runtime_error);
		}
	}

	TEST_CASE("MaxPool2D") {
		SUBCASE("MaxPool2D Calculation") {
			Tensor<float> input{ {1, 1, 6, 6} };
			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool2D(input, 2, 2);

			CHECK(output.GetShape() == Shape(1, 1, 3, 3));

			CHECK(output[0] == 8);
			CHECK(output[1] == 10);
			CHECK(output[2] == 12);
			CHECK(output[3] == 20);
			CHECK(output[4] == 22);
			CHECK(output[5] == 24);
			CHECK(output[6] == 32);
			CHECK(output[7] == 34);
			CHECK(output[8] == 36);
		}

		SUBCASE("MaxPool2D calculates correctly with multiple input and output channels") {
			Tensor<float> input{ {1, 2, 4, 4} };
			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool2D(input, 2, 2);

			CHECK(output.GetShape() == Shape(1, 2, 2, 2));

			CHECK(output[0] == 6);
			CHECK(output[1] == 8);
			CHECK(output[2] == 14);
			CHECK(output[3] == 16);
			CHECK(output[4] == 22);
			CHECK(output[5] == 24);
			CHECK(output[6] == 30);
			CHECK(output[7] == 32);
		}

		SUBCASE("MaxPool2D with ceilMode gives correct shape") {
			Tensor<float> input{ {1, 1, 7, 7} };
			input.Fill(1.0f);

			auto output = MaxPool2D(input, 2, 2, 0, 0, 0, 0, 1, 1, true);

			CHECK(output.GetShape() == Shape(1, 1, 4, 4));

			for (auto& val : output) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("MaxPool2D strides change output dimensions") {
			auto input = Tensor<float>::Ones({ 1, 1, 6, 6});

			auto output = MaxPool2D(input, 2, 2, 1, 1);

			CHECK(output.GetShape() == Shape(1, 1, 5, 5));

			for (auto& val : output) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("MaxPool2D padding correctly maintains dimensions") {
			auto input = Tensor<float>::Ones({ 1, 1, 6, 6 });

			auto output = MaxPool2D(input, 2, 2, 0, 0, 1, 1);

			CHECK(output.GetShape() == Shape(1, 1, 4, 4));

			for (auto& val : output) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("MaxPool2D dilation correctly maintains dimensions") {
			auto input = Tensor<float>::Ones({ 1, 1, 6, 6 });

			auto output = MaxPool2D(input, 2, 2, 1, 1, 0, 0, 2, 2);

			CHECK(output.GetShape() == Shape(1, 1, 4, 4));

			for (auto& val : output) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("MaxPool2D rejects invalid input dimensions") {
			auto input = Tensor<float>::Ones({ 1, 1, 2 });

			CHECK_THROWS_AS(MaxPool2D(input, 1, 1), std::runtime_error);
		}
	}

	TEST_CASE("MaxPool3D") {
		SUBCASE("MaxPool3D Calculation") {
			Tensor<float> input{ {1, 1, 4, 4, 4} };
			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool3D(input, 2, 2, 2);

			CHECK(output.GetShape() == Shape(1, 1, 2, 2, 2));

			CHECK(output[0] == 22);
			CHECK(output[1] == 24);
			CHECK(output[2] == 30);
			CHECK(output[3] == 32);
			CHECK(output[4] == 54);
			CHECK(output[5] == 56);
			CHECK(output[6] == 62);
			CHECK(output[7] == 64);
		}

		SUBCASE("MaxPool3D calculates correctly with multiple input and output channels") {
			Tensor<float> input{ {1, 2, 4, 4, 4} };
			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool3D(input, 2, 2, 2);

			CHECK(output.GetShape() == Shape(1, 2, 2, 2, 2));

			CHECK(output[0] == 22);
			CHECK(output[1] == 24);
			CHECK(output[2] == 30);
			CHECK(output[3] == 32);
			CHECK(output[4] == 54);
			CHECK(output[5] == 56);
			CHECK(output[6] == 62);
			CHECK(output[7] == 64);
			CHECK(output[8] == 86);
			CHECK(output[9] == 88);
			CHECK(output[10] == 94);
			CHECK(output[11] == 96);
			CHECK(output[12] == 118);
			CHECK(output[13] == 120);
			CHECK(output[14] == 126);
			CHECK(output[15] == 128);
		}

		SUBCASE("MaxPool3D with ceilMode gives correct shape") {
			Tensor<float> input{ {1, 1, 7, 7, 7} };
			input.Fill(1.0f);

			auto output = MaxPool3D(input, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, true);

			CHECK(output.GetShape() == Shape(1, 1, 4, 4, 4));

			for (auto& val : output) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("MaxPool3D strides change output dimensions") {
			auto input = Tensor<float>::Ones({ 1, 1, 4, 4, 4 });

			auto output = MaxPool3D(input, 2, 2, 2, 1, 1, 1);

			CHECK(output.GetShape() == Shape(1, 1, 3, 3, 3));

			for (auto& val : output) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("MaxPool3D padding correctly maintains dimensions") {
			auto input = Tensor<float>::Ones({ 1, 1, 4, 4, 4 });

			auto output = MaxPool3D(input, 2, 2, 2, 0, 0, 0, 1, 1, 1);

			CHECK(output.GetShape() == Shape(1, 1, 3, 3, 3));

			for (auto& val : output) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("MaxPool3D dilation correctly maintains dimensions") {
			auto input = Tensor<float>::Ones({ 1, 1, 4, 4, 4 });

			auto output = MaxPool3D(input, 2, 2, 2, 1, 1, 1, 0, 0, 0, 2, 2, 2);

			CHECK(output.GetShape() == Shape(1, 1, 2, 2, 2));

			for (auto& val : output) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("MaxPool3D rejects invalid input dimensions") {
			auto input = Tensor<float>::Ones({ 1, 1, 2, 2 });

			CHECK_THROWS_AS(MaxPool3D(input, 1, 1, 1), std::runtime_error);
		}
	}
}