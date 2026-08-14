/// convolutionalTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/convolution/convolution.h>

using namespace MLCore::Utils;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Convolutional Operation Tests") {
    TEST_CASE("Conv1D") {
        SUBCASE("Conv1D Calculation") {
            Tensor<float> input{ {1, 1, 3} };
            Tensor<float> kernel{ {1, 1, 2} };

            input[0] = 1.0f;
            input[1] = 2.0f;
            input[2] = 3.0f;

            kernel[0] = 1.0f;
            kernel[1] = 0.0f;

            auto bias = Tensor<float>::Zeros({ 1 });

            auto output = Conv1D(input, kernel, &bias);

            CHECK(output.GetShape() == Shape(1, 1, 2));

            CHECK(output[0] == 1.0f);
            CHECK(output[1] == 2.0f);
        }

        SUBCASE("Conv1D bias correctly affects output tensor") {
            Tensor<float> input{ {1, 1, 2} };
            input[0] = 3;
            input[1] = 4;

            auto kernel = Tensor<float>::Custom({ 1, 1, 1 }, 2.0f);
            auto bias = Tensor<float>::Custom({ 1 }, 5.0f);

            auto output = Conv1D(input, kernel, &bias);

            CHECK(output.GetShape() == Shape(1, 1, 2));

            CHECK(output[0] == 11.0f);
            CHECK(output[1] == 13.0f);
        }

        SUBCASE("Conv1D calculates correctly without bias") {
            Tensor<float> input{ {1, 1, 2} };
            input[0] = 2;
            input[1] = 3;

            auto kernel = Tensor<float>::Custom({ 1, 1, 1 }, 2.0f);

            auto output = Conv1D(input, kernel);

            CHECK(output[0] == 4.0f);
            CHECK(output[1] == 6.0f);
        }

        SUBCASE("Conv1D calculates correctly with multiple input and output channels") {
            Tensor<float> input{ {1, 2, 2} };
            Tensor<float> kernel{ {2, 2, 1} };

            /// First input channel
            input[0] = 1.0f;
            input[1] = 2.0f;

            /// Second input channel
            input[2] = 3.0f;
            input[3] = 4.0f;

            /// First output channel
            kernel[0] = 1.0f;
            kernel[1] = 2.0f;

            /// Second output channel
            kernel[2] = 3.0f;
            kernel[3] = 4.0f;

            auto output = Conv1D(input, kernel);

            CHECK(output.GetShape() == Shape(1, 2, 2));

            CHECK(output[0] == 7.0f);
            CHECK(output[1] == 10.0f);
            CHECK(output[2] == 15.0f);
            CHECK(output[3] == 22.0f);
        }

        SUBCASE("Conv1D stride changes output dimensions") {
            auto input = Tensor<float>::Ones({ 1, 1, 4 });
            auto kernel = Tensor<float>::Ones({ 1, 1, 2 });

            auto output = Conv1D<float>(input, kernel, nullptr, 2);

            CHECK(output.GetShape() == Shape(1, 1, 2));

            for (auto& val : output) {
                CHECK(val == 2.0f);
            }
        }

        SUBCASE("Conv1D padding correctly maintains dimensions") {
            auto input = Tensor<float>::Ones({ 1, 1, 2 });
            auto kernel = Tensor<float>::Ones({ 1, 1, 3 });

            auto output = Conv1D<float>(input, kernel, nullptr, 1, 1);

            CHECK(output.GetShape() == Shape(1, 1, 2));

            for (auto& val : output) {
                CHECK(val == 2.0f);
            }
        }

        SUBCASE("Conv1D dilation correctly maintains dimensions") {
            auto input = Tensor<float>::Ones({ 1, 1, 5 });
            auto kernel = Tensor<float>::Ones({ 1, 1, 2 });

            auto output = Conv1D<float>(input, kernel, nullptr, 1, 0, 2);

            CHECK(output.GetShape() == Shape(1, 1, 3));

            for (auto& val : output) {
                CHECK(val == 2.0f);
            }
        }

        SUBCASE("Conv1D rejects invalid input rank") {
            Tensor<float> input{ {3, 3} };
            Tensor<float> weight{ {1, 1, 2} };

            CHECK_THROWS_AS(Conv1D(input, weight), std::runtime_error);
        }

        SUBCASE("Conv1D rejects invalid kernel rank") {
            Tensor<float> input{ {1, 1, 3} };
            Tensor<float> weight{ {2, 2} };

            CHECK_THROWS_AS(Conv1D(input, weight), std::runtime_error);
        }

        SUBCASE("Conv1D rejects mimatched channels") {
            Tensor<float> input{ {1, 2, 3} };
            Tensor<float> weight{ {1, 1, 2} };

            CHECK_THROWS_AS(Conv1D(input, weight), std::runtime_error);
        }
    }

	TEST_CASE("Conv2D") {
		SUBCASE("Conv2D Calculation") {
			Tensor<float> input{ {1, 1, 3, 3} };
			Tensor<float> kernel{ {1, 1, 2, 2} };

            input[0] = 1.0f;
            input[1] = 2.0f;
            input[2] = 3.0f;
            input[3] = 4.0f;
            input[4] = 5.0f;
            input[5] = 6.0f;
            input[6] = 7.0f;
            input[7] = 8.0f;
            input[8] = 9.0f;

            kernel[0] = 1.0f;
            kernel[1] = 0.0f;
            kernel[2] = 0.0f;
            kernel[3] = 1.0f;

            auto bias = Tensor<float>::Zeros({ 1 });

            auto output = Conv2D(input, kernel, &bias);

            CHECK(output.GetShape() == Shape(1, 1, 2, 2));

            CHECK(output[0] == 6.0f);
            CHECK(output[1] == 8.0f);
            CHECK(output[2] == 12.0f);
            CHECK(output[3] == 14.0f);
		}

        SUBCASE("Conv2D bias correctly affects output tensor") {
            Tensor<float> input{ {1, 1, 2, 2} };
            input[0] = 1;
            input[1] = 2;
            input[2] = 3;
            input[3] = 4;

            auto kernel = Tensor<float>::Custom({ 1, 1, 1, 1 }, 2.0f);
            auto bias = Tensor<float>::Custom({ 1 }, 5.0f);

            auto output = Conv2D(input, kernel, &bias);

            CHECK(output.GetShape() == Shape(1, 1, 2, 2));

            CHECK(output[0] == 7.0f);
            CHECK(output[1] == 9.0f);
            CHECK(output[2] == 11.0f);
            CHECK(output[3] == 13.0f);
        }

        SUBCASE("Conv2D calculates correctly without bias") {
            Tensor<float> input{ {1, 1, 2, 2} };
            input[0] = 1;
            input[1] = 2;
            input[2] = 3;
            input[3] = 4;

            auto kernel = Tensor<float>::Custom({ 1, 1, 1, 1 }, 2.0f);

            auto output = Conv2D(input, kernel);

            CHECK(output[0] == 2.0f);
            CHECK(output[1] == 4.0f);
            CHECK(output[2] == 6.0f);
            CHECK(output[3] == 8.0f);
        }

        SUBCASE("Conv2D calculates correctly with multiple input and output channels") {
            Tensor<float> input{ {1, 2, 2, 2} };
            Tensor<float> kernel{ {2, 2, 1, 1} };

            /// First input channel
            input[0] = 1.0f;
            input[1] = 2.0f;
            input[2] = 3.0f;
            input[3] = 4.0f;

            /// Second input channel
            input[4] = 5.0f;
            input[5] = 6.0f;
            input[6] = 7.0f;
            input[7] = 8.0f;

            /// First output channel
            kernel[0] = 1.0f;
            kernel[1] = 2.0f;

            /// Second output channel
            kernel[2] = 3.0f;
            kernel[3] = 4.0f;

            auto output = Conv2D(input, kernel);

            CHECK(output.GetShape() == Shape(1, 2, 2, 2));

            CHECK(output[0] == 11.0f);
            CHECK(output[1] == 14.0f);
            CHECK(output[2] == 17.0f);
            CHECK(output[3] == 20.0f);

            CHECK(output[4] == 23.0f);
            CHECK(output[5] == 30.0f);
            CHECK(output[6] == 37.0f);
            CHECK(output[7] == 44.0f);
        }

        SUBCASE("Conv2D stride changes output dimensions") {
            auto input = Tensor<float>::Ones({ 1, 1, 4, 4 });
            auto kernel = Tensor<float>::Ones({ 1, 1, 2, 2 });

            auto output = Conv2D<float>(input, kernel, nullptr, 2, 2);

            CHECK(output.GetShape() == Shape(1, 1, 2, 2));

            for (auto& val : output) {
                CHECK(val == 4.0f);
            }
        }

        SUBCASE("Conv2D padding correctly maintains dimensions") {
            auto input = Tensor<float>::Ones({ 1, 1, 2, 2 });
            auto kernel = Tensor<float>::Ones({ 1, 1, 3, 3 });

            auto output = Conv2D<float>(input, kernel, nullptr, 1, 1, 1, 1);

            CHECK(output.GetShape() == Shape(1, 1, 2, 2));

            for (auto& val : output) {
                CHECK(val == 4.0f);
            }
        }

        SUBCASE("Conv2D dilation correctly maintains dimensions") {
            auto input = Tensor<float>::Ones({ 1, 1, 5, 5 });
            auto kernel = Tensor<float>::Ones({ 1, 1, 2, 2 });

            auto output = Conv2D<float>(input, kernel, nullptr, 1, 1, 0, 0, 2, 2);

            CHECK(output.GetShape() == Shape(1, 1, 3, 3));

            for (auto& val : output) {
                CHECK(val == 4.0f);
            }
        }

        SUBCASE("Conv2D rejects invalid input rank") {
            Tensor<float> input{ {3, 3} };
            Tensor<float> weight{ {1, 1, 2, 2} };

            CHECK_THROWS_AS(Conv2D<float>(input, weight), std::runtime_error);
        }

        SUBCASE("Conv2D rejects invalid kernel rank") {
            Tensor<float> input{ {1, 1, 3, 3} };
            Tensor<float> weight{ {2, 2} };

            CHECK_THROWS_AS(Conv2D<float>(input, weight), std::runtime_error);
        }

        SUBCASE("Conv2D rejects mimatched channels") {
            Tensor<float> input{ {1, 2, 3, 3} };
            Tensor<float> weight{ {1, 1, 2, 2} };

            CHECK_THROWS_AS(Conv2D<float>(input, weight), std::runtime_error);
        }
	}

    TEST_CASE("Conv3D") {
        SUBCASE("Conv3D Calculation") {
            auto input = Tensor<float>::Ones({ 1, 1, 3, 3, 2 });
            auto kernel = Tensor<float>::Ones({ 1, 1, 2, 2, 2 });

            auto bias = Tensor<float>::Zeros({ 1 });

            auto output = Conv3D(input, kernel, &bias);

            CHECK(output.GetShape() == Shape(1, 1, 2, 2, 1));

            for (auto& val : output) {
                CHECK(val == 8.0f);
            }
        }

        SUBCASE("Conv3D bias correctly affects output tensor") {
            Tensor<float> input{ {1, 1, 2, 2, 2} };
            input[0] = 1;
            input[1] = 2;
            input[2] = 3;
            input[3] = 4;
            input[4] = 5;
            input[5] = 6;
            input[6] = 7;
            input[7] = 8;

            auto kernel = Tensor<float>::Custom({ 1, 1, 1, 1, 1 }, 2.0f);
            auto bias = Tensor<float>::Custom({ 1 }, 5.0f);

            auto output = Conv3D(input, kernel, &bias);

            CHECK(output.GetShape() == Shape(1, 1, 2, 2, 2));

            CHECK(output[0] == 7.0f);
            CHECK(output[1] == 9.0f);
            CHECK(output[2] == 11.0f);
            CHECK(output[3] == 13.0f);
            CHECK(output[4] == 15.0f);
            CHECK(output[5] == 17.0f);
            CHECK(output[6] == 19.0f);
            CHECK(output[7] == 21.0f);
        }

        SUBCASE("Conv3D calculates correctly without bias") {
            Tensor<float> input{ {1, 1, 2, 2, 2} };
            input[0] = 1;
            input[1] = 2;
            input[2] = 3;
            input[3] = 4;
            input[4] = 1;
            input[5] = 2;
            input[6] = 3;
            input[7] = 4;

            auto kernel = Tensor<float>::Custom({ 1, 1, 1, 1, 1 }, 2.0f);

            auto output = Conv3D(input, kernel);

            CHECK(output[0] == 2.0f);
            CHECK(output[1] == 4.0f);
            CHECK(output[2] == 6.0f);
            CHECK(output[3] == 8.0f);
            CHECK(output[4] == 2.0f);
            CHECK(output[5] == 4.0f);
            CHECK(output[6] == 6.0f);
            CHECK(output[7] == 8.0f);
        }

        SUBCASE("Conv3D calculates correctly with multiple input and output channels") {
            Tensor<float> input{ {1, 2, 2, 2, 1} };
            Tensor<float> kernel{ {2, 2, 1, 1, 1 } };

            /// First input channel
            input[0] = 1.0f;
            input[1] = 2.0f;
            input[2] = 3.0f;
            input[3] = 4.0f;

            /// Second input channel
            input[4] = 5.0f;
            input[5] = 6.0f;
            input[6] = 7.0f;
            input[7] = 8.0f;

            /// First output channel
            kernel[0] = 1.0f;
            kernel[1] = 2.0f;

            /// Second output channel
            kernel[2] = 3.0f;
            kernel[3] = 4.0f;

            auto output = Conv3D(input, kernel);

            CHECK(output.GetShape() == Shape(1, 2, 2, 2, 1));

            CHECK(output[0] == 11.0f);
            CHECK(output[1] == 14.0f);
            CHECK(output[2] == 17.0f);
            CHECK(output[3] == 20.0f);

            CHECK(output[4] == 23.0f);
            CHECK(output[5] == 30.0f);
            CHECK(output[6] == 37.0f);
            CHECK(output[7] == 44.0f);
        }

        SUBCASE("Conv3D stride changes output dimensions") {
            auto input = Tensor<float>::Ones({ 1, 1, 4, 4, 1 });
            auto kernel = Tensor<float>::Ones({ 1, 1, 2, 2, 1 });

            auto output = Conv3D<float>(input, kernel, nullptr, 2, 2, 2);

            CHECK(output.GetShape() == Shape(1, 1, 2, 2, 1));

            for (auto& val : output) {
                CHECK(val == 4.0f);
            }
        }

        SUBCASE("Conv3D padding correctly maintains dimensions") {
            auto input = Tensor<float>::Ones({ 1, 1, 2, 2, 1 });
            auto kernel = Tensor<float>::Ones({ 1, 1, 3, 3, 1 });

            auto output = Conv3D<float>(input, kernel, nullptr, 1, 1, 1, 1, 1, 1);

            CHECK(output.GetShape() == Shape(1, 1, 2, 2, 3));

            CHECK(output[0] == 0.0f);
            CHECK(output[1] == 4.0f);
            CHECK(output[2] == 0.0f);
            CHECK(output[3] == 0.0f);
            CHECK(output[4] == 4.0f);
            CHECK(output[5] == 0.0f);
            CHECK(output[6] == 0.0f);
            CHECK(output[7] == 4.0f);
            CHECK(output[8] == 0.0f);
            CHECK(output[9] == 0.0f);
            CHECK(output[10] == 4.0f);
            CHECK(output[11] == 0.0f);
        }

        SUBCASE("Conv3D dilation correctly maintains dimensions") {
            auto input = Tensor<float>::Ones({ 1, 1, 5, 5, 1 });
            auto kernel = Tensor<float>::Ones({ 1, 1, 2, 2, 1 });

            auto output = Conv3D<float>(input, kernel, nullptr, 1, 1, 1, 0, 0, 0, 2, 2, 2);

            CHECK(output.GetShape() == Shape(1, 1, 3, 3, 1));

            for (auto& val : output) {
                CHECK(val == 4.0f);
            }
        }

        SUBCASE("Conv3D rejects invalid input rank") {
            Tensor<float> input{ {3, 3} };
            Tensor<float> weight{ {1, 1, 2, 2, 2} };

            CHECK_THROWS_AS(Conv3D<float>(input, weight), std::runtime_error);
        }

        SUBCASE("Conv3D rejects invalid kernel rank") {
            Tensor<float> input{ {1, 1, 3, 3, 3} };
            Tensor<float> weight{ {2, 2} };

            CHECK_THROWS_AS(Conv3D<float>(input, weight), std::runtime_error);
        }

        SUBCASE("Conv3D rejects mimatched channels") {
            Tensor<float> input{ {1, 2, 3, 3, 3} };
            Tensor<float> weight{ {1, 1, 2, 2, 2} };

            CHECK_THROWS_AS(Conv3D<float>(input, weight), std::runtime_error);
        }
    }
}