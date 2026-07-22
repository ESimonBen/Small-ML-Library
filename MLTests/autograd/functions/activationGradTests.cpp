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
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = -3;
			A.SetRequiresGrad(true);

			auto B = ReLU(A, allocator);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B, allocator);
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
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(ReLU(A, allocator), std::runtime_error);
		}
	}

	TEST_CASE("LeakyReLU Gradient") {
		SUBCASE("LeakyReLU Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = -3;
			A.SetRequiresGrad(true);

			auto B = LeakyReLU(A, 0.1f, allocator);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B, allocator);
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
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(LeakyReLU(A, 0.1f, allocator), std::runtime_error);
		}
	}

	TEST_CASE("Sigmoid Gradient") {
		SUBCASE("Sigmoid Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = -4;
			A.SetRequiresGrad(true);

			auto B = Sigmoid(A, allocator);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B, allocator);

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
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Sigmoid(A, allocator), std::runtime_error);
		}
	}

	TEST_CASE("Tanh Gradient") {
		SUBCASE("Tanh Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = -4;
			A.SetRequiresGrad(true);

			auto B = Tanh(A, allocator);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B, allocator);
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
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Tanh(A, allocator), std::runtime_error);
		}
	}

	TEST_CASE("Softmax Gradient") {
		SUBCASE("Softmax Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = 4;
			A.SetRequiresGrad(true);

			auto B = Softmax(A, allocator);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B, allocator);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 0.0f);
			}
		}

		SUBCASE("Empty tensor Softmax throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Softmax(A, allocator), std::runtime_error);
		}
	}

	TEST_CASE("AxisSoftmax Gradient") {
		SUBCASE("AxisSoftmax Gradient Operation (axis 0)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = 4;
			A.SetRequiresGrad(true);

			auto B = AxisSoftmax(A, 0, allocator);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B, allocator);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 0.0f);
			}
		}

		SUBCASE("AxisSoftmax Gradient Operation (axis 1)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = 4;
			A.SetRequiresGrad(true);

			auto B = AxisSoftmax(A, 1, allocator);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B, allocator);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 0.0f);
			}
		}

		SUBCASE("Empty tensor AxisSoftmax throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(AxisSoftmax(A, 0, allocator), std::out_of_range);
		}
	}

	TEST_CASE("AxisLogSoftmax Gradient") {
		SUBCASE("AxisLogSoftmax Gradient Operation (axis 0)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = 4;
			A.SetRequiresGrad(true);

			auto B = AxisLogSoftmax(A, 0, allocator);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B, allocator);
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

		SUBCASE("Empty tensor AxisLogSoftMax throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(AxisLogSoftmax(A, 0, allocator), std::out_of_range);
		}
	}
}