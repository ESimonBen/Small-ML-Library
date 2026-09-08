/// poolingGradTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/pooling/pooling.h>

using namespace MLCore::Utils;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Pooling Gradient Tests") {
	TEST_CASE("MaxPool1D Gradient") {
		SUBCASE("MaxPool1D Gradient Operation (Kernel Size 1)") {
			Tensor<float> input{ {1, 1, 6} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool1D(input, 1);

			CHECK(output.GetShape() == Shape(1, 1, 6));

			auto gradient = Tensor<float>::Ones({ 1, 1, 6 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 6));

			CHECK(grad[0] == 1);
			CHECK(grad[1] == 1);
			CHECK(grad[2] == 1);
			CHECK(grad[3] == 1);
			CHECK(grad[4] == 1);
			CHECK(grad[5] == 1);
		}

		SUBCASE("MaxPool1D Gradient Operation (Kernel Size 2)") {
			Tensor<float> input{ {1, 1, 6} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool1D(input, 2);

			CHECK(output.GetShape() == Shape(1, 1, 3));

			auto gradient = Tensor<float>::Ones({ 1, 1, 3 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 6));

			CHECK(grad[0] == 0);
			CHECK(grad[1] == 1);
			CHECK(grad[2] == 0);
			CHECK(grad[3] == 1);
			CHECK(grad[4] == 0);
			CHECK(grad[5] == 1);
		}

		SUBCASE("MaxPool1D Gradient Operation (Multiple Channels)") {
			Tensor<float> input{ {1, 2, 6} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool1D(input, 2);

			CHECK(output.GetShape() == Shape(1, 2, 3));

			auto gradient = Tensor<float>::Ones({ 1, 2, 3 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 2, 6));

			CHECK(grad[0] == 0);
			CHECK(grad[1] == 1);
			CHECK(grad[2] == 0);
			CHECK(grad[3] == 1);
			CHECK(grad[4] == 0);
			CHECK(grad[5] == 1);
			CHECK(grad[6] == 0);
			CHECK(grad[7] == 1);
			CHECK(grad[8] == 0);
			CHECK(grad[9] == 1);
			CHECK(grad[10] == 0);
			CHECK(grad[11] == 1);
		}

		SUBCASE("MaxPool1D Gradient Operation (CeilMode Enabled)") {
			Tensor<float> input{ {1, 1, 7} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool1D(input, 2, 0, 0, 1, true);

			CHECK(output.GetShape() == Shape(1, 1, 4));

			auto gradient = Tensor<float>::Ones({1, 1, 4});
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 7));

			CHECK(grad[0] == 0);
			CHECK(grad[1] == 1);
			CHECK(grad[2] == 0);
			CHECK(grad[3] == 1);
			CHECK(grad[4] == 0);
			CHECK(grad[5] == 1);
			CHECK(grad[6] == 1);
		}

		SUBCASE("MaxPool1D Gradient Operation (With Custom Stride)") {
			Tensor<float> input{ {1, 1, 6} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool1D(input, 2, 1);

			CHECK(output.GetShape() == Shape(1, 1, 5));

			auto gradient = Tensor<float>::Ones({ 1, 1, 5 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 6));

			CHECK(grad[0] == 0);
			CHECK(grad[1] == 1);
			CHECK(grad[2] == 1);
			CHECK(grad[3] == 1);
			CHECK(grad[4] == 1);
			CHECK(grad[5] == 1);
		}

		SUBCASE("MaxPool1D Gradient Operation (With Padding)") {
			Tensor<float> input{ {1, 1, 6} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool1D(input, 2, 0, 1);

			CHECK(output.GetShape() == Shape(1, 1, 4));

			auto gradient = Tensor<float>::Ones({ 1, 1, 4 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 6));

			CHECK(grad[0] == 1);
			CHECK(grad[1] == 0);
			CHECK(grad[2] == 1);
			CHECK(grad[3] == 0);
			CHECK(grad[4] == 1);
			CHECK(grad[5] == 1);
		}

		SUBCASE("MaxPool1D Gradient Operation (With Dilation)") {
			Tensor<float> input{ {1, 1, 6} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool1D(input, 2, 1, 0, 2);

			CHECK(output.GetShape() == Shape(1, 1, 4));

			auto gradient = Tensor<float>::Ones({ 1, 1, 4 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 6));

			CHECK(grad[0] == 0);
			CHECK(grad[1] == 0);
			CHECK(grad[2] == 1);
			CHECK(grad[3] == 1);
			CHECK(grad[4] == 1);
			CHECK(grad[5] == 1);
		}
	}

	TEST_CASE("MaxPool2D Gradient") {
		SUBCASE("MaxPool2D Gradient Operation (Kernel Size 1)") {
			Tensor<float> input{ {1, 1, 4, 4} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool2D(input, 1, 1);

			CHECK(output.GetShape() == Shape(1, 1, 4, 4));

			auto gradient = Tensor<float>::Ones({ 1, 1, 4, 4 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 4, 4));

			CHECK(grad[0] == 1);
			CHECK(grad[1] == 1);
			CHECK(grad[2] == 1);
			CHECK(grad[3] == 1);
			CHECK(grad[4] == 1);
			CHECK(grad[5] == 1);
			CHECK(grad[6] == 1);
			CHECK(grad[7] == 1);
			CHECK(grad[8] == 1);
			CHECK(grad[9] == 1);
			CHECK(grad[10] == 1);
			CHECK(grad[11] == 1);
			CHECK(grad[12] == 1);
			CHECK(grad[13] == 1);
			CHECK(grad[14] == 1);
			CHECK(grad[15] == 1);
		}

		SUBCASE("MaxPool2D Gradient Operation (Kernel Size 2)") {
			Tensor<float> input{ {1, 1, 4, 4} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool2D(input, 2, 2);

			CHECK(output.GetShape() == Shape(1, 1, 2, 2));

			auto gradient = Tensor<float>::Ones({ 1, 1, 2, 2 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 4, 4));

			CHECK(grad[0] == 0);
			CHECK(grad[1] == 0);
			CHECK(grad[2] == 0);
			CHECK(grad[3] == 0);
			CHECK(grad[4] == 0);
			CHECK(grad[5] == 1);
			CHECK(grad[6] == 0);
			CHECK(grad[7] == 1);
			CHECK(grad[8] == 0);
			CHECK(grad[9] == 0);
			CHECK(grad[10] == 0);
			CHECK(grad[11] == 0);
			CHECK(grad[12] == 0);
			CHECK(grad[13] == 1);
			CHECK(grad[14] == 0);
			CHECK(grad[15] == 1);
		}

		SUBCASE("MaxPool2D Gradient Operation (Multiple Channels)") {
			Tensor<float> input{ {1, 2, 4, 4} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool2D(input, 2, 2);

			CHECK(output.GetShape() == Shape(1, 2, 2, 2));

			auto gradient = Tensor<float>::Ones({ 1, 2, 2, 2 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 2, 4, 4));

			CHECK(grad[0] == 0);
			CHECK(grad[1] == 0);
			CHECK(grad[2] == 0);
			CHECK(grad[3] == 0);
			CHECK(grad[4] == 0);
			CHECK(grad[5] == 1);
			CHECK(grad[6] == 0);
			CHECK(grad[7] == 1);
			CHECK(grad[8] == 0);
			CHECK(grad[9] == 0);
			CHECK(grad[10] == 0);
			CHECK(grad[11] == 0);
			CHECK(grad[12] == 0);
			CHECK(grad[13] == 1);
			CHECK(grad[14] == 0);
			CHECK(grad[15] == 1);
			CHECK(grad[16] == 0);
			CHECK(grad[17] == 0);
			CHECK(grad[18] == 0);
			CHECK(grad[19] == 0);
			CHECK(grad[20] == 0);
			CHECK(grad[21] == 1);
			CHECK(grad[22] == 0);
			CHECK(grad[23] == 1);
			CHECK(grad[24] == 0);
			CHECK(grad[25] == 0);
			CHECK(grad[26] == 0);
			CHECK(grad[27] == 0);
			CHECK(grad[28] == 0);
			CHECK(grad[29] == 1);
			CHECK(grad[30] == 0);
			CHECK(grad[31] == 1);
		}

		SUBCASE("MaxPool2D Gradient Operation (CeilMode Enabled)") {
			Tensor<float> input{ {1, 1, 3, 3} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool2D(input, 2, 2, 0, 0, 0, 0, 1, 1, true);

			CHECK(output.GetShape() == Shape(1, 1, 2, 2));

			auto gradient = Tensor<float>::Ones({ 1, 1, 2, 2 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 3, 3));

			CHECK(grad[0] == 0);
			CHECK(grad[1] == 0);
			CHECK(grad[2] == 0);
			CHECK(grad[3] == 0);
			CHECK(grad[4] == 1);
			CHECK(grad[5] == 1);
			CHECK(grad[6] == 0);
			CHECK(grad[7] == 1);
			CHECK(grad[8] == 1);
		}

		SUBCASE("MaxPool2D Gradient Operation (With Custom Stride)") {
			Tensor<float> input{ {1, 1, 4, 4} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool2D(input, 2, 2, 1, 1);

			CHECK(output.GetShape() == Shape(1, 1, 3, 3));

			auto gradient = Tensor<float>::Ones({ 1, 1, 3, 3 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 4, 4));

			CHECK(grad[0] == 0);
			CHECK(grad[1] == 0);
			CHECK(grad[2] == 0);
			CHECK(grad[3] == 0);
			CHECK(grad[4] == 0);
			CHECK(grad[5] == 1);
			CHECK(grad[6] == 1);
			CHECK(grad[7] == 1);
			CHECK(grad[8] == 0);
			CHECK(grad[9] == 1);
			CHECK(grad[10] == 1);
			CHECK(grad[11] == 1);
			CHECK(grad[12] == 0);
			CHECK(grad[13] == 1);
			CHECK(grad[14] == 1);
			CHECK(grad[15] == 1);
		}

		SUBCASE("MaxPool2D Gradient Operation (With Padding)") {
			Tensor<float> input{ {1, 1, 4, 4} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool2D(input, 2, 2, 0, 0, 1, 1);

			CHECK(output.GetShape() == Shape(1, 1, 3, 3));

			auto gradient = Tensor<float>::Ones({ 1, 1, 3, 3 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 4, 4));

			CHECK(grad[0] == 1);
			CHECK(grad[1] == 0);
			CHECK(grad[2] == 1);
			CHECK(grad[3] == 1);
			CHECK(grad[4] == 0);
			CHECK(grad[5] == 0);
		}

		SUBCASE("MaxPool2D Gradient Operation (With Dilation)") {
			Tensor<float> input{ {1, 1, 6, 6} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool2D(input, 2, 2, 1, 1, 0, 0, 2, 2);

			CHECK(output.GetShape() == Shape(1, 1, 4, 4));

			auto gradient = Tensor<float>::Ones({ 1, 1, 4, 4 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 6, 6));

			CHECK(grad[0] == 0);
			CHECK(grad[1] == 0);
			CHECK(grad[2] == 0);
			CHECK(grad[3] == 0);
			CHECK(grad[4] == 0);
			CHECK(grad[5] == 0);
			CHECK(grad[6] == 0);
			CHECK(grad[7] == 0);
			CHECK(grad[8] == 0);
			CHECK(grad[9] == 0);
			CHECK(grad[10] == 0);
			CHECK(grad[11] == 0);
			CHECK(grad[12] == 0);
			CHECK(grad[13] == 0);
			CHECK(grad[14] == 1);
			CHECK(grad[15] == 1);
			CHECK(grad[16] == 1);
			CHECK(grad[17] == 1);
			CHECK(grad[18] == 0);
			CHECK(grad[19] == 0);
			CHECK(grad[20] == 1);
			CHECK(grad[21] == 1);
			CHECK(grad[22] == 1);
			CHECK(grad[23] == 1);
			CHECK(grad[24] == 0);
			CHECK(grad[25] == 0);
			CHECK(grad[26] == 1);
			CHECK(grad[27] == 1);
			CHECK(grad[28] == 1);
			CHECK(grad[29] == 1);
			CHECK(grad[30] == 0);
			CHECK(grad[31] == 0);
			CHECK(grad[32] == 1);
			CHECK(grad[33] == 1);
			CHECK(grad[34] == 1);
			CHECK(grad[35] == 1);
		}
	}

	TEST_CASE("MaxPool3D Gradient") {
		SUBCASE("MaxPool3D Gradient Operation (Kernel Size 1)") {
			Tensor<float> input{ {1, 1, 2, 2, 2} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool3D(input, 1, 1, 1);

			CHECK(output.GetShape() == Shape(1, 1, 2, 2, 2));

			auto gradient = Tensor<float>::Ones({ 1, 1, 2, 2, 2 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 2, 2, 2));

			CHECK(grad[0] == 1);
			CHECK(grad[1] == 1);
			CHECK(grad[2] == 1);
			CHECK(grad[3] == 1);
			CHECK(grad[4] == 1);
			CHECK(grad[5] == 1);
			CHECK(grad[6] == 1);
			CHECK(grad[7] == 1);
		}

		SUBCASE("MaxPool3D Gradient Operation (Kernel Size 2)") {
			Tensor<float> input{ {1, 1, 2, 4, 4} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool3D(input, 2, 2, 2);

			CHECK(output.GetShape() == Shape(1, 1, 1, 2, 2));

			auto gradient = Tensor<float>::Ones({ 1, 1, 1, 2, 2 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 2, 4, 4));

			for (size_t i = 0; i < 32; ++i) {
				if (i == 21 || i == 23 || i == 29 || i == 31) {
					CHECK(grad[i] == 1);
				}
				else {
					CHECK(grad[i] == 0);
				}
			}
		}

		SUBCASE("MaxPool3D Gradient Operation (Multiple Channels)") {
			Tensor<float> input{ {1, 2, 2, 4, 4} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool3D(input, 2, 2, 2);

			CHECK(output.GetShape() == Shape(1, 2, 1, 2, 2));

			auto gradient = Tensor<float>::Ones({ 1, 2, 1, 2, 2 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 2, 2, 4, 4));

			for (size_t i = 0; i < 64; ++i) {
				if (i == 21 || i == 23 || i == 29 || i == 31 ||
					i == 53 || i == 55 || i == 61 || i == 63) {
					CHECK(grad[i] == 1);
				}
				else {
					CHECK(grad[i] == 0);
				}
			}
		}

		SUBCASE("MaxPool3D Gradient Operation (CeilMode Enabled)") {
			Tensor<float> input{ {1, 1, 2, 3, 3} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool3D(input, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, true);

			CHECK(output.GetShape() == Shape(1, 1, 1, 2, 2));

			auto gradient = Tensor<float>::Ones({ 1, 1, 1, 2, 2 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 2, 3, 3));

			for (size_t i = 0; i < 18; ++i) {
				if (i == 13 || i == 14 || i == 16 || i == 17) {
					CHECK(grad[i] == 1);
				}
				else {
					CHECK(grad[i] == 0);
				}
			}
		}

		SUBCASE("MaxPool3D Gradient Operation (With Custom Stride)") {
			Tensor<float> input{ {1, 1, 2, 4, 4} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool3D(input, 2, 2, 2, 1, 1, 1);

			CHECK(output.GetShape() == Shape(1, 1, 1, 3, 3));

			auto gradient = Tensor<float>::Ones({ 1, 1, 1, 3, 3 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 2, 4, 4));

			CHECK(grad[21] == 1); CHECK(grad[22] == 1); CHECK(grad[23] == 1);
			CHECK(grad[25] == 1); CHECK(grad[26] == 1); CHECK(grad[27] == 1);
			CHECK(grad[29] == 1); CHECK(grad[30] == 1); CHECK(grad[31] == 1);
		}

		SUBCASE("MaxPool3D Gradient Operation (With Padding)") {
			Tensor<float> input{ {1, 1, 2, 4, 4} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool3D(input, 2, 2, 2, 0, 0, 0, 1, 1, 1);

			CHECK(output.GetShape() == Shape(1, 1, 2, 3, 3));

			auto gradient = Tensor<float>::Ones({ 1, 1, 2, 3, 3 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 2, 4, 4));

			CHECK(grad[0] == 1);
			CHECK(grad[1] == 0);
			CHECK(grad[2] == 1);
			CHECK(grad[3] == 1);
		}

		SUBCASE("MaxPool3D Gradient Operation (With Dilation)") {
			Tensor<float> input{ {1, 1, 4, 6, 6} };
			input.SetRequiresGrad(true);

			size_t size = input.NumElements();
			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			auto output = MaxPool3D(input, 2, 2, 2, 1, 1, 1, 0, 0, 0, 2, 2, 2);

			CHECK(output.GetShape() == Shape(1, 1, 2, 4, 4));

			auto gradient = Tensor<float>::Ones({ 1, 1, 2, 4, 4 });
			output.Backward(gradient);

			auto grad = input.Grad();

			CHECK(grad.GetShape() == Shape(1, 1, 4, 6, 6));

			CHECK(grad[0] == 0);
			CHECK(grad[72] == 0); 
			CHECK(grad[108] == 0);
		}
	}

}