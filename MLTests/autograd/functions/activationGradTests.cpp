/// activationGradTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/activations/activation.h>

using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Activation Function Gradient Tests") {
	TEST_CASE("ReLU Gradient") {
		SUBCASE("ReLU Gradient Operation") {
			Tensor<float> A({ 2, 3 });
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = -3;
			A.SetRequiresGrad(true);

			auto B = ReLU(A);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			
			size_t size = gradA.NumElements();
			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == 1.0f);
				}
				else {
					CHECK(gradA[i] == 0.0f);
				}
			}
		}

		SUBCASE("Empty tensor ReLU throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(ReLU(A), std::runtime_error);
		}
	}

	TEST_CASE("LeakyReLU Gradient") {
		SUBCASE("LeakyReLU Gradient Operation") {
			Tensor<float> A({ 2, 3 });
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = -3;
			A.SetRequiresGrad(true);

			auto B = LeakyReLU(A, 0.1f);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements();
			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == 1.0f);
				}
				else {
					CHECK(gradA[i] == doctest::Approx(0.1f));
				}
			}
		}

		SUBCASE("Empty tensor LeakyReLU throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(LeakyReLU(A, 0.1f), std::runtime_error);
		}
	}

	TEST_CASE("Sigmoid Gradient") {
		SUBCASE("Sigmoid Gradient Operation") {
			Tensor<float> A({ 2, 3 });
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = -4;
			A.SetRequiresGrad(true);

			auto B = Sigmoid(A);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);

			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements();
			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(0.0451767f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(0.0176627f));
				}
			}
		}

		SUBCASE("Empty tensor Sigmoid throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Sigmoid(A), std::runtime_error);
		}
	}

	TEST_CASE("Tanh Gradient") {
		SUBCASE("Tanh Gradient Operation") {
			Tensor<float> A({ 2, 3 });
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = -4;
			A.SetRequiresGrad(true);

			auto B = Tanh(A);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements();
			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(0.00986603f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(0.00134099f));
				}
			}
		}

		SUBCASE("Empty tensor Tanh throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Tanh(A), std::runtime_error);
		}
	}

	TEST_CASE("Softmax Gradient") {
		SUBCASE("Softmax Gradient Operation") {
			Tensor<float> A({ 2, 3 });
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = 4;
			A.SetRequiresGrad(true);

			auto B = Softmax(A);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 0.0f);
			}
		}

		SUBCASE("Empty tensor Softmax throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Softmax(A), std::runtime_error);
		}
	}

	TEST_CASE("AxisSoftmax Gradient") {
		SUBCASE("AxisSoftmax Gradient Operation (axis 0)") {
			Tensor<float> A({ 2, 3 });
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = 4;
			A.SetRequiresGrad(true);

			auto B = AxisSoftmax(A, 0);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 0.0f);
			}
		}

		SUBCASE("AxisSoftmax Gradient Operation (axis 1)") {
			Tensor<float> A({ 2, 3 });
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = 4;
			A.SetRequiresGrad(true);

			auto B = AxisSoftmax(A, 1);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 0.0f);
			}
		}

		SUBCASE("Empty tensor AxisSoftmax throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(AxisSoftmax(A, 0), std::out_of_range);
		}
	}

	TEST_CASE("AxisLogSoftmax Gradient") {
		SUBCASE("AxisLogSoftmax Gradient Operation (axis 0)") {
			Tensor<float> A({ 2, 3 });
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = 4;
			A.SetRequiresGrad(true);

			auto B = AxisLogSoftmax(A, 0);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements();
			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(0.462117f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(-0.462117f));
				}
			}
		}

		SUBCASE("AxisLogSoftmax Gradient Operation (axis 1)") {
			Tensor<float> A({ 2, 3 });
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = 4;
			A.SetRequiresGrad(true);

			auto B = AxisLogSoftmax(A, 1);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 0.0f);
			}
		}

		SUBCASE("Empty tensor AxisLogSoftMax throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(AxisLogSoftmax(A, 0), std::out_of_range);
		}
	}
}