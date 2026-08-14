/// convolutionGradTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/operations.h>

using namespace MLCore::Utils;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Convolution Gradient Tests") {
	TEST_CASE("Conv1D Gradient") {
		SUBCASE("Conv1D Gradient Operation (Kernel Length 1)") {
			Tensor<float> input{ {1, 1, 2} };
			input[0] = 2.0f;
			input[1] = 3.0f;
			input.SetRequiresGrad(true);

			auto kernel = Tensor<float>::Custom({ 1, 1, 1 }, 4.0f);
			kernel.SetRequiresGrad(true);

			auto bias = Tensor<float>::Ones({ 1 });
			bias.SetRequiresGrad(true);

			auto output = Conv1D(input, kernel, &bias);

			CHECK(output.GetShape() == Shape(1, 1, 2));

			auto loss = SumAll(output);
			loss.Backward();

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();
			auto gradBias = bias.Grad();

			CHECK(gradInput[0] == 4.0f);
			CHECK(gradInput[1] == 4.0f);
			CHECK(gradKernel[0] == 5.0f);
			CHECK(gradBias[0] == 2.0f);
		}

		SUBCASE("Conv1D Gradient Operation (Kernel Length 2)") {
			Tensor<float> input{ {1, 1, 3} };
			Tensor<float> kernel{ {1, 1, 2} };

			size_t inputSize = input.NumElements();
			size_t kernelSize = kernel.NumElements();

			for (size_t i = 0; i < inputSize; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			for (size_t i = 0; i < kernelSize; ++i) {
				kernel[i] = static_cast<float>(i + 1);
			}

			input.SetRequiresGrad(true);
			kernel.SetRequiresGrad(true);

			auto output = Conv1D(input, kernel);

			CHECK(output.GetShape() == Shape(1, 1, 2));

			auto gradOutput = Tensor<float>::Ones({ 1, 1, 2 });
			output.Backward(gradOutput);

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();

			CHECK(gradInput[0] == 1.0f);
			CHECK(gradInput[1] == 3.0f);
			CHECK(gradInput[2] == 2.0f);

			CHECK(gradKernel[0] == 3.0f);
			CHECK(gradKernel[1] == 5.0f);
		}

		SUBCASE("Conv1D Gradient Operation (Multiple Channels)") {
			Tensor<float> input{ {1, 2, 2} };
			Tensor<float> kernel{ {2, 2, 1} };

			/// Channel 0
			input[0] = 1.0f;
			input[1] = 2.0f;

			/// Channel 1
			input[2] = 3.0f;
			input[3] = 4.0f;

			kernel[0] = 1.0f;
			kernel[1] = 2.0f;
			kernel[2] = 3.0f;
			kernel[3] = 4.0f;

			input.SetRequiresGrad(true);
			kernel.SetRequiresGrad(true);

			auto output = Conv1D(input, kernel);

			CHECK(output.GetShape() == Shape(1, 2, 2));

			auto gradOutput = Tensor<float>::Ones({ 1, 2, 2 });
			output.Backward(gradOutput);

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();

			CHECK(gradInput[0] == 4.0f);
			CHECK(gradInput[1] == 4.0f);
			CHECK(gradInput[2] == 6.0f);
			CHECK(gradInput[3] == 6.0f);

			CHECK(gradKernel[0] == 3.0f);
			CHECK(gradKernel[1] == 7.0f);
			CHECK(gradKernel[2] == 3.0f);
			CHECK(gradKernel[3] == 7.0f);
		}

		SUBCASE("Conv1D Gradient Operation (With Stride)") {
			Tensor<float> input{ {1, 1, 4} };
			size_t size = input.NumElements();

			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			input.SetRequiresGrad(true);

			auto kernel = Tensor<float>::Ones({ 1, 1, 2 });
			kernel.SetRequiresGrad(true);

			auto output = Conv1D<float>(input, kernel, nullptr, 2);

			CHECK(output.GetShape() == Shape(1, 1, 2));

			auto gradOutput = Tensor<float>::Ones({ 1, 1, 2 });
			output.Backward(gradOutput);

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();

			for (auto& val : gradInput) {
				CHECK(val == 1.0f);
			}

			CHECK(gradKernel[0] == 4.0f);
			CHECK(gradKernel[1] == 6.0f);
		}

		SUBCASE("Conv1D Gradient Operation (With Padding)") {
			Tensor<float> input{ {1, 1, 2} };
			size_t size = input.NumElements();

			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			input.SetRequiresGrad(true);

			auto kernel = Tensor<float>::Ones({ 1, 1, 3 });
			kernel.SetRequiresGrad(true);

			auto output = Conv1D<float>(input, kernel, nullptr, 1, 1);

			CHECK(output.GetShape() == Shape(1, 1, 2));

			auto gradOutput = Tensor<float>::Ones({ 1, 1, 2 });
			output.Backward(gradOutput);

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();

			for (auto& val : gradInput) {
				CHECK(val == 2.0f);
			}

			CHECK(gradKernel[0] == 1.0f);
			CHECK(gradKernel[1] == 3.0f);
			CHECK(gradKernel[2] == 2.0f);
		}

		SUBCASE("Conv1D Gradient Operation (With Dilation)") {
			Tensor<float> input{ {1, 1, 5} };
			size_t size = input.NumElements();

			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			input.SetRequiresGrad(true);

			auto kernel = Tensor<float>::Ones({ 1, 1, 2 });
			kernel.SetRequiresGrad(true);

			auto output = Conv1D<float>(input, kernel, nullptr, 1, 0, 2);

			CHECK(output.GetShape() == Shape(1, 1, 3));

			auto gradOutput = Tensor<float>::Ones({ 1, 1, 3 });
			output.Backward(gradOutput);

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();

			CHECK(gradInput[0] == 1.0f);
			CHECK(gradInput[1] == 1.0f);
			CHECK(gradInput[2] == 2.0f);
			CHECK(gradInput[3] == 1.0f);
			CHECK(gradInput[4] == 1.0f);

			CHECK(gradKernel[0] == 6.0f);
			CHECK(gradKernel[1] == 12.0f);
		}
	}

	TEST_CASE("Conv2D Gradient") {
		SUBCASE("Conv2D Gradient Operation (1x1 Kernel)") {
			Tensor<float> input{ {1, 1, 1, 2} };
			input[0] = 2.0f;
			input[1] = 3.0f;
			input.SetRequiresGrad(true);
			
			auto kernel = Tensor<float>::Custom({ 1, 1, 1, 1 }, 4.0f);
			kernel.SetRequiresGrad(true);

			auto bias = Tensor<float>::Ones({ 1 });
			bias.SetRequiresGrad(true);

			auto output = Conv2D(input, kernel, &bias);

			CHECK(output.GetShape() == Shape(1, 1, 1, 2));

			auto loss = SumAll(output);
			loss.Backward();

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();
			auto gradBias = bias.Grad();

			CHECK(gradInput[0] == 4.0f);
			CHECK(gradInput[1] == 4.0f);
			CHECK(gradKernel[0] == 5.0f);
			CHECK(gradBias[0] == 2.0f);
		}

		SUBCASE("Conv2D Gradient Operation (2x2 Kernel)") {
			Tensor<float> input{ {1, 1, 3, 3} };
			Tensor<float> kernel{ {1, 1, 2, 2} };

			size_t inputSize = input.NumElements();
			size_t kernelSize = kernel.NumElements();

			for (size_t i = 0; i < inputSize; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			for (size_t i = 0; i < kernelSize; ++i) {
				kernel[i] = static_cast<float>(i + 1);
			}

			input.SetRequiresGrad(true);
			kernel.SetRequiresGrad(true);

			auto output = Conv2D(input, kernel);

			CHECK(output.GetShape() == Shape(1, 1, 2, 2));

			auto gradOutput = Tensor<float>::Ones({ 1, 1, 2, 2 });
			output.Backward(gradOutput);

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();

			CHECK(gradInput[0] == 1.0f);
			CHECK(gradInput[1] == 3.0f);
			CHECK(gradInput[2] == 2.0f);
			CHECK(gradInput[3] == 4.0f);
			CHECK(gradInput[4] == 10.0f);
			CHECK(gradInput[5] == 6.0f);
			CHECK(gradInput[6] == 3.0f);
			CHECK(gradInput[7] == 7.0f);
			CHECK(gradInput[8] == 4.0f);

			CHECK(gradKernel[0] == 12.0f);
			CHECK(gradKernel[1] == 16.0f);
			CHECK(gradKernel[2] == 24.0f);
			CHECK(gradKernel[3] == 28.0f);
		}

		SUBCASE("Conv2D Gradient Operation (Multiple Channels)") {
			Tensor<float> input{ {1, 2, 2, 2} };
			Tensor<float> kernel{ {2, 2, 1, 1} };

			/// Channel 0
			input[0] = 1.0f;
			input[1] = 2.0f;
			input[2] = 3.0f;
			input[3] = 4.0f;

			/// Channel 1
			input[4] = 5.0f;
			input[5] = 6.0f;
			input[6] = 7.0f;
			input[7] = 8.0f;

			kernel[0] = 1.0f;
			kernel[1] = 2.0f;
			kernel[2] = 3.0f;
			kernel[3] = 4.0f;

			input.SetRequiresGrad(true);
			kernel.SetRequiresGrad(true);

			auto output = Conv2D(input, kernel);

			CHECK(output.GetShape() == Shape(1, 2, 2, 2));

			auto gradOutput = Tensor<float>::Ones({ 1, 2, 2, 2 });
			output.Backward(gradOutput);

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();

			CHECK(gradInput[0] == 4.0f);
			CHECK(gradInput[1] == 4.0f);
			CHECK(gradInput[2] == 4.0f);
			CHECK(gradInput[3] == 4.0f);
			CHECK(gradInput[4] == 6.0f);
			CHECK(gradInput[5] == 6.0f);
			CHECK(gradInput[6] == 6.0f);
			CHECK(gradInput[7] == 6.0f);

			CHECK(gradKernel[0] == 10.0f);
			CHECK(gradKernel[1] == 26.0f);
			CHECK(gradKernel[2] == 10.0f);
			CHECK(gradKernel[3] == 26.0f);
		}

		SUBCASE("Conv2D Gradient Operation (With Stride)") {
			Tensor<float> input{ {1, 1, 4, 4} };
			size_t size = input.NumElements();

			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			input.SetRequiresGrad(true);

			auto kernel = Tensor<float>::Ones({ 1, 1, 2, 2 });
			kernel.SetRequiresGrad(true);

			auto output = Conv2D<float>(input, kernel, nullptr, 2, 2);

			CHECK(output.GetShape() == Shape(1, 1, 2, 2));

			auto gradOutput = Tensor<float>::Ones({ 1, 1, 2, 2 });
			output.Backward(gradOutput);

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();

			for (auto& val : gradInput) {
				CHECK(val == 1.0f);
			}

			CHECK(gradKernel[0] == 24.0f);
			CHECK(gradKernel[1] == 28.0f);
			CHECK(gradKernel[2] == 40.0f);
			CHECK(gradKernel[3] == 44.0f);
		}

		SUBCASE("Conv2D Gradient Operation (With Padding)") {
			Tensor<float> input{ {1, 1, 2, 2} };
			size_t size = input.NumElements();

			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			input.SetRequiresGrad(true);

			auto kernel = Tensor<float>::Ones({ 1, 1, 3, 3 });
			kernel.SetRequiresGrad(true);
			
			auto output = Conv2D<float>(input, kernel, nullptr, 1, 1, 1, 1);

			CHECK(output.GetShape() == Shape(1, 1, 2, 2));

			auto gradOutput = Tensor<float>::Ones({ 1, 1, 2, 2 });
			output.Backward(gradOutput);

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();

			for (auto& val : gradInput) {
				CHECK(val == 4.0f);
			}

			CHECK(gradKernel[0] == 1.0f);
			CHECK(gradKernel[1] == 3.0f);
			CHECK(gradKernel[2] == 2.0f);
			CHECK(gradKernel[3] == 4.0f);
			CHECK(gradKernel[4] == 10.0f);
			CHECK(gradKernel[5] == 6.0f);
			CHECK(gradKernel[6] == 3.0f);
			CHECK(gradKernel[7] == 7.0f);
			CHECK(gradKernel[8] == 4.0f);
		}

		SUBCASE("Conv2D Gradient Operation (With Dilation)") {
			Tensor<float> input{ {1, 1, 5, 5} };
			size_t size = input.NumElements();

			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			input.SetRequiresGrad(true);

			auto kernel = Tensor<float>::Ones({ 1, 1, 2, 2 });
			kernel.SetRequiresGrad(true);

			auto output = Conv2D<float>(input, kernel, nullptr, 1, 1, 0, 0, 2, 2);

			CHECK(output.GetShape() == Shape(1, 1, 3, 3));

			auto gradOutput = Tensor<float>::Ones({ 1, 1, 3, 3 });
			output.Backward(gradOutput);

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();

			CHECK(gradInput[0] == 1.0f);
			CHECK(gradInput[1] == 1.0f);
			CHECK(gradInput[2] == 2.0f);
			CHECK(gradInput[3] == 1.0f);
			CHECK(gradInput[4] == 1.0f);
			CHECK(gradInput[5] == 1.0f);
			CHECK(gradInput[6] == 1.0f);
			CHECK(gradInput[7] == 2.0f);
			CHECK(gradInput[8] == 1.0f);
			CHECK(gradInput[9] == 1.0f);
			CHECK(gradInput[10] == 2.0f);
			CHECK(gradInput[11] == 2.0f);
			CHECK(gradInput[12] == 4.0f);
			CHECK(gradInput[13] == 2.0f);
			CHECK(gradInput[14] == 2.0f);
			CHECK(gradInput[15] == 1.0f);
			CHECK(gradInput[16] == 1.0f);
			CHECK(gradInput[17] == 2.0f);
			CHECK(gradInput[18] == 1.0f);
			CHECK(gradInput[19] == 1.0f);

			CHECK(gradKernel[0] == 63.0f);
			CHECK(gradKernel[1] == 81.0f);
			CHECK(gradKernel[2] == 153.0f);
			CHECK(gradKernel[3] == 171.0f);
		}
	}

	TEST_CASE("Conv3D Gradient") {
		SUBCASE("Conv3D Gradient Operation (1x1x1 Kernel)") {
			Tensor<float> input{ {1, 1, 1, 2, 1} };
			input[0] = 2.0f;
			input[1] = 3.0f;
			input.SetRequiresGrad(true);

			auto kernel = Tensor<float>::Custom({ 1, 1, 1, 1, 1 }, 4.0f);
			kernel.SetRequiresGrad(true);

			auto bias = Tensor<float>::Ones({ 1 });
			bias.SetRequiresGrad(true);

			auto output = Conv3D(input, kernel, &bias);

			CHECK(output.GetShape() == Shape(1, 1, 1, 2, 1));

			auto loss = SumAll(output);
			loss.Backward();

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();
			auto gradBias = bias.Grad();

			CHECK(gradInput[0] == 4.0f);
			CHECK(gradInput[1] == 4.0f);
			CHECK(gradKernel[0] == 5.0f);
			CHECK(gradBias[0] == 2.0f);
		}

		SUBCASE("Conv3D Gradient Operation (2x2x2 Kernel)") {
			Tensor<float> input{ {1, 1, 3, 3, 3} };
			Tensor<float> kernel{ {1, 1, 2, 2, 2} };

			size_t inputSize = input.NumElements();
			size_t kernelSize = kernel.NumElements();

			for (size_t i = 0; i < inputSize; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			for (size_t i = 0; i < kernelSize; ++i) {
				kernel[i] = static_cast<float>(i + 1);
			}

			input.SetRequiresGrad(true);
			kernel.SetRequiresGrad(true);

			auto output = Conv3D(input, kernel);

			CHECK(output.GetShape() == Shape(1, 1, 2, 2, 2));

			auto gradOutput = Tensor<float>::Ones({ 1, 1, 2, 2, 2 });
			output.Backward(gradOutput);

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();

			CHECK(gradInput[0] == 1.0f);
			CHECK(gradInput[1] == 3.0f);
			CHECK(gradInput[2] == 2.0f);
			CHECK(gradInput[3] == 4.0f);
			CHECK(gradInput[4] == 10.0f);
			CHECK(gradInput[5] == 6.0f);
			CHECK(gradInput[6] == 3.0f);
			CHECK(gradInput[7] == 7.0f);
			CHECK(gradInput[8] == 4.0f);
			CHECK(gradInput[9] == 6.0f);
			CHECK(gradInput[10] == 14.0f);
			CHECK(gradInput[11] == 8.0f);
			CHECK(gradInput[12] == 16.0f);
			CHECK(gradInput[13] == 36.0f);
			CHECK(gradInput[14] == 20.0f);
			CHECK(gradInput[15] == 10.0f);
			CHECK(gradInput[16] == 22.0f);
			CHECK(gradInput[17] == 12.0f);

			CHECK(gradInput[18] == 5.0f);
			CHECK(gradInput[19] == 11.0f);
			CHECK(gradInput[20] == 6.0f);
			CHECK(gradInput[21] == 12.0f);
			CHECK(gradInput[22] == 26.0f);
			CHECK(gradInput[23] == 14.0f);
			CHECK(gradInput[24] == 7.0f);
			CHECK(gradInput[25] == 15.0f);
			CHECK(gradInput[26] == 8.0f);

			CHECK(gradKernel[0] == 60.0f);
			CHECK(gradKernel[1] == 68.0f);
			CHECK(gradKernel[2] == 84.0f);
			CHECK(gradKernel[3] == 92.0f);
			CHECK(gradKernel[4] == 132.0f);
			CHECK(gradKernel[5] == 140.0f);
			CHECK(gradKernel[6] == 156.0f);
			CHECK(gradKernel[7] == 164.0f);

		}

		SUBCASE("Conv3D Gradient Operation (Multiple Channels)") {
			Tensor<float> input{ {1, 2, 2, 2, 1} };
			Tensor<float> kernel{ {2, 2, 1, 1, 1} };

			/// Channel 0
			input[0] = 1.0f;
			input[1] = 2.0f;
			input[2] = 3.0f;
			input[3] = 4.0f;

			/// Channel 1
			input[4] = 5.0f;
			input[5] = 6.0f;
			input[6] = 7.0f;
			input[7] = 8.0f;

			kernel[0] = 1.0f;
			kernel[1] = 2.0f;
			kernel[2] = 3.0f;
			kernel[3] = 4.0f;

			input.SetRequiresGrad(true);
			kernel.SetRequiresGrad(true);

			auto output = Conv3D(input, kernel);

			CHECK(output.GetShape() == Shape(1, 2, 2, 2, 1));

			auto gradOutput = Tensor<float>::Ones({ 1, 2, 2, 2, 1 });
			output.Backward(gradOutput);

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();

			CHECK(gradInput[0] == 4.0f);
			CHECK(gradInput[1] == 4.0f);
			CHECK(gradInput[2] == 4.0f);
			CHECK(gradInput[3] == 4.0f);
			CHECK(gradInput[4] == 6.0f);
			CHECK(gradInput[5] == 6.0f);
			CHECK(gradInput[6] == 6.0f);
			CHECK(gradInput[7] == 6.0f);

			CHECK(gradKernel[0] == 10.0f);
			CHECK(gradKernel[1] == 26.0f);
			CHECK(gradKernel[2] == 10.0f);
			CHECK(gradKernel[3] == 26.0f);
		}

		SUBCASE("Conv3D Gradient Operation (With Stride)") {
			Tensor<float> input{ {1, 1, 4, 4, 1} };
			size_t size = input.NumElements();

			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			input.SetRequiresGrad(true);

			auto kernel = Tensor<float>::Ones({ 1, 1, 2, 2, 1 });
			kernel.SetRequiresGrad(true);

			auto output = Conv3D<float>(input, kernel, nullptr, 2, 2, 2);

			CHECK(output.GetShape() == Shape(1, 1, 2, 2, 1));

			auto gradOutput = Tensor<float>::Ones({ 1, 1, 2, 2, 1 });
			output.Backward(gradOutput);

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();

			for (auto& val : gradInput) {
				CHECK(val == 1.0f);
			}

			CHECK(gradKernel[0] == 24.0f);
			CHECK(gradKernel[1] == 28.0f);
			CHECK(gradKernel[2] == 40.0f);
			CHECK(gradKernel[3] == 44.0f);
		}

		SUBCASE("Conv3D Gradient Operation (With Padding)") {
			Tensor<float> input{ {1, 1, 2, 2, 1} };
			size_t size = input.NumElements();

			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			input.SetRequiresGrad(true);

			auto kernel = Tensor<float>::Ones({ 1, 1, 3, 3, 1 });
			kernel.SetRequiresGrad(true);

			auto output = Conv3D<float>(input, kernel, nullptr, 1, 1, 1, 1, 1, 1);

			CHECK(output.GetShape() == Shape(1, 1, 2, 2, 3));

			auto gradOutput = Tensor<float>::Ones({ 1, 1, 2, 2, 3 });
			output.Backward(gradOutput);

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();

			for (auto& val : gradInput) {
				CHECK(val == 4.0f);
			}

			CHECK(gradKernel[0] == 1.0f);
			CHECK(gradKernel[1] == 3.0f);
			CHECK(gradKernel[2] == 2.0f);
			CHECK(gradKernel[3] == 4.0f);
			CHECK(gradKernel[4] == 10.0f);
			CHECK(gradKernel[5] == 6.0f);
			CHECK(gradKernel[6] == 3.0f);
			CHECK(gradKernel[7] == 7.0f);
			CHECK(gradKernel[8] == 4.0f);
		}

		SUBCASE("Conv3D Gradient Operation (With Dilation)") {
			Tensor<float> input{ {1, 1, 5, 5, 1} };
			size_t size = input.NumElements();

			for (size_t i = 0; i < size; ++i) {
				input[i] = static_cast<float>(i + 1);
			}

			input.SetRequiresGrad(true);

			auto kernel = Tensor<float>::Ones({ 1, 1, 2, 2, 1 });
			kernel.SetRequiresGrad(true);

			auto output = Conv3D<float>(input, kernel, nullptr, 1, 1, 1, 0, 0, 0, 2, 2, 2);

			CHECK(output.GetShape() == Shape(1, 1, 3, 3, 1));

			auto gradOutput = Tensor<float>::Ones({ 1, 1, 3, 3, 1 });
			output.Backward(gradOutput);

			auto gradInput = input.Grad();
			auto gradKernel = kernel.Grad();

			CHECK(gradInput[0] == 1.0f);
			CHECK(gradInput[1] == 1.0f);
			CHECK(gradInput[2] == 2.0f);
			CHECK(gradInput[3] == 1.0f);
			CHECK(gradInput[4] == 1.0f);
			CHECK(gradInput[5] == 1.0f);
			CHECK(gradInput[6] == 1.0f);
			CHECK(gradInput[7] == 2.0f);
			CHECK(gradInput[8] == 1.0f);
			CHECK(gradInput[9] == 1.0f);
			CHECK(gradInput[10] == 2.0f);
			CHECK(gradInput[11] == 2.0f);
			CHECK(gradInput[12] == 4.0f);
			CHECK(gradInput[13] == 2.0f);
			CHECK(gradInput[14] == 2.0f);
			CHECK(gradInput[15] == 1.0f);
			CHECK(gradInput[16] == 1.0f);
			CHECK(gradInput[17] == 2.0f);
			CHECK(gradInput[18] == 1.0f);
			CHECK(gradInput[19] == 1.0f);

			CHECK(gradKernel[0] == 63.0f);
			CHECK(gradKernel[1] == 81.0f);
			CHECK(gradKernel[2] == 153.0f);
			CHECK(gradKernel[3] == 171.0f);
		}
	}
}