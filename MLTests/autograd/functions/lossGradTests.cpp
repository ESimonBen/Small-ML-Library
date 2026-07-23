/// lossGradTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/loss/loss.h>

using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Loss Function Gradient Tests") {
	TEST_CASE("MeanSquaredError Gradient") {
		SUBCASE("MeanSquaredError Gradient Operation (No Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3.0f;
			A[3] = A[4] = A[5] = 5.0f;
			B.Fill(4.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = MeanSquaredError(A, B, Reduction::None, allocator);
			CHECK(C.GetShape() == Shape(2, 1));
			CHECK(C.RequiresGrad());

			auto loss = SumAll(C, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements(); /// The sizes of the gradients of A and B are the same

			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(-0.666667f));
					CHECK(gradB[i] == doctest::Approx(0.666667f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(0.666667f));
					CHECK(gradB[i] == doctest::Approx(-0.666667f));
				}
			}
		}

		SUBCASE("MeanSquaredError Gradient Operation (SumAll Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3.0f;
			A[3] = A[4] = A[5] = 5.0f;
			B.Fill(4.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = MeanSquaredError(A, B, Reduction::Sum, allocator);
			CHECK(C.GetShape() == Shape(1));
			CHECK(C.RequiresGrad());

			C.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements(); /// The sizes of the gradients of A and B are the same

			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(-0.666667f));
					CHECK(gradB[i] == doctest::Approx(0.666667f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(0.666667f));
					CHECK(gradB[i] == doctest::Approx(-0.666667f));
				}
			}
		}

		SUBCASE("MeanSquaredError Gradient Operation (MeanAll Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3.0f;
			A[3] = A[4] = A[5] = 5.0f;
			B.Fill(4.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = MeanSquaredError(A, B, Reduction::Mean, allocator);
			CHECK(C.GetShape() == Shape(1));
			CHECK(C.RequiresGrad());

			C.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements(); /// The sizes of the gradients of A and B are the same

			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(-0.333333f));
					CHECK(gradB[i] == doctest::Approx(0.333333f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(0.333333f));
					CHECK(gradB[i] == doctest::Approx(-0.333333f));
				}
			}
		}

		SUBCASE("Empty tensor MeanSquaredError throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			Tensor<float> B(Shape(), allocator);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			CHECK_THROWS_AS(MeanSquaredError(A, B, Reduction::None, allocator), std::runtime_error);
		}
	}

	TEST_CASE("MeanAbsoluteError Gradient") {
		SUBCASE("MeanAbsoluteError Gradient Operation (No Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3.0f;
			A[3] = A[4] = A[5] = 5.0f;
			B.Fill(4.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = MeanAbsoluteError(A, B, Reduction::None, allocator);
			CHECK(C.GetShape() == Shape(2, 1));
			CHECK(C.RequiresGrad());

			auto loss = SumAll(C, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements(); /// The sizes of the gradients of A and B are the same

			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(-0.333333f));
					CHECK(gradB[i] == doctest::Approx(0.333333f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(0.333333f));
					CHECK(gradB[i] == doctest::Approx(-0.333333f));
				}
			}
		}

		SUBCASE("MeanAbsoluteError Gradient Operation (SumAll Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3.0f;
			A[3] = A[4] = A[5] = 5.0f;
			B.Fill(4.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = MeanAbsoluteError(A, B, Reduction::Sum, allocator);
			CHECK(C.GetShape() == Shape(1));
			CHECK(C.RequiresGrad());

			C.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements(); /// The sizes of the gradients of A and B are the same

			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(-0.333333f));
					CHECK(gradB[i] == doctest::Approx(0.333333f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(0.333333f));
					CHECK(gradB[i] == doctest::Approx(-0.333333f));
				}
			}
		}

		SUBCASE("MeanAbsoluteError Gradient Operation (MeanAll Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3.0f;
			A[3] = A[4] = A[5] = 5.0f;
			B.Fill(4.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = MeanAbsoluteError(A, B, Reduction::Mean, allocator);
			CHECK(C.GetShape() == Shape(1));
			CHECK(C.RequiresGrad());

			C.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements(); /// The sizes of the gradients of A and B are the same

			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(-0.166667f));
					CHECK(gradB[i] == doctest::Approx(0.166667f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(0.166667f));
					CHECK(gradB[i] == doctest::Approx(-0.166667f));
				}
			}
		}

		SUBCASE("Empty tensor MeanAbsoluteError throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			Tensor<float> B(Shape(), allocator);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			CHECK_THROWS_AS(MeanAbsoluteError(A, B, Reduction::None, allocator), std::runtime_error);
		}
	}

	TEST_CASE("BinaryCrossEntropy Gradient") {
		SUBCASE("BinaryCrossEntropy Gradient Operation (No Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 0.3f;
			A[3] = A[4] = A[5] = 0.5f;
			B.Fill(0.4f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = BinaryCrossEntropy(A, B, Reduction::None, allocator);
			CHECK(C.GetShape() == Shape(2, 1));
			CHECK(C.RequiresGrad());

			auto loss = SumAll(C, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements(); /// The sizes of the gradients of A and B are the same

			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(-0.15873f));
					CHECK(gradB[i] == doctest::Approx(0.282433f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(0.133333f));
					CHECK(gradB[i] == doctest::Approx(0.0f));
				}
			}
		}

		SUBCASE("BinaryCrossEntropy Gradient Operation (SumAll Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 0.3f;
			A[3] = A[4] = A[5] = 0.5f;
			B.Fill(0.4f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = BinaryCrossEntropy(A, B, Reduction::Sum, allocator);
			CHECK(C.GetShape() == Shape(1));
			CHECK(C.RequiresGrad());

			C.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements(); /// The sizes of the gradients of A and B are the same

			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(-0.15873f));
					CHECK(gradB[i] == doctest::Approx(0.282433f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(0.133333f));
					CHECK(gradB[i] == doctest::Approx(0.0f));
				}
			}
		}

		SUBCASE("BinaryCrossEntropy Gradient Operation (MeanAll Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 0.3f;
			A[3] = A[4] = A[5] = 0.5f;
			B.Fill(0.4f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = BinaryCrossEntropy(A, B, Reduction::Mean, allocator);
			CHECK(C.GetShape() == Shape(1));
			CHECK(C.RequiresGrad());

			C.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements(); /// The sizes of the gradients of A and B are the same

			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(-0.0793651f));
					CHECK(gradB[i] == doctest::Approx(0.141216f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(0.0666667f));
					CHECK(gradB[i] == doctest::Approx(0.0f));
				}
			}
		}

		SUBCASE("Empty tensor BinaryCrossEntropy throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			Tensor<float> B(Shape(), allocator);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			CHECK_THROWS_AS(BinaryCrossEntropy(A, B, Reduction::None, allocator), std::runtime_error);
		}
	}

	TEST_CASE("BinaryCrossEntropyWithLogits Gradient") {
		SUBCASE("BinaryCrossEntropyWithLogits Gradient Operation (No Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3.0f;
			A[3] = A[4] = A[5] = 5.0f;
			B.Fill(4.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = BinaryCrossEntropyWithLogits(A, B, Reduction::None, allocator);
			CHECK(C.GetShape() == Shape(2, 1));
			CHECK(C.RequiresGrad());

			auto loss = SumAll(C, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements(); /// The sizes of the gradients of A and B are the same

			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(-1.01581f));
					CHECK(gradB[i] == doctest::Approx(-1.0f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(-1.00223f));
					CHECK(gradB[i] == doctest::Approx(-1.66667f));
				}
			}
		}

		SUBCASE("BinaryCrossEntropyWithLogits Gradient Operation (SumAll Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3.0f;
			A[3] = A[4] = A[5] = 5.0f;
			B.Fill(4.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = BinaryCrossEntropyWithLogits(A, B, Reduction::Sum, allocator);
			CHECK(C.GetShape() == Shape(1));
			CHECK(C.RequiresGrad());

			C.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements(); /// The sizes of the gradients of A and B are the same

			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(-1.01581f));
					CHECK(gradB[i] == doctest::Approx(-1.0f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(-1.00223f));
					CHECK(gradB[i] == doctest::Approx(-1.66667f));
				}
			}
		}

		SUBCASE("BinaryCrossEntropyWithLogits Gradient Operation (MeanAll Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3.0f;
			A[3] = A[4] = A[5] = 5.0f;
			B.Fill(4.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = BinaryCrossEntropyWithLogits(A, B, Reduction::Mean, allocator);
			CHECK(C.GetShape() == Shape(1));
			CHECK(C.RequiresGrad());

			C.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements(); /// The sizes of the gradients of A and B are the same

			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(-0.507904f));
					CHECK(gradB[i] == doctest::Approx(-0.5f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(-0.501116f));
					CHECK(gradB[i] == doctest::Approx(-0.833333f));
				}
			}
		}

		SUBCASE("Empty tensor BinaryCrossEntropyWithLogits throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			Tensor<float> B(Shape(), allocator);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			CHECK_THROWS_AS(BinaryCrossEntropyWithLogits(A, B, Reduction::None, allocator), std::runtime_error);
		}
	}

	TEST_CASE("CrossEntropy Gradient") {
		SUBCASE("CrossEntropy Gradient Operation (No Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 0.3f;
			A[3] = A[4] = A[5] = 0.5f;
			B.Fill(0.4f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = CrossEntropy(A, B, Reduction::None, allocator);
			CHECK(C.GetShape() == Shape(2, 1));
			CHECK(C.RequiresGrad());

			auto loss = SumAll(C, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements(); /// The sizes of the gradients of A and B are the same

			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(-0.444444f));
					CHECK(gradB[i] == doctest::Approx(0.401324f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(-0.266667f));
					CHECK(gradB[i] == doctest::Approx(0.231049f));
				}
			}
		}

		SUBCASE("CrossEntropy Gradient Operation (SumAll Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 0.3f;
			A[3] = A[4] = A[5] = 0.5f;
			B.Fill(0.4f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = CrossEntropy(A, B, Reduction::Sum, allocator);
			CHECK(C.GetShape() == Shape(1));
			CHECK(C.RequiresGrad());

			C.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements(); /// The sizes of the gradients of A and B are the same

			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(-0.444444f));
					CHECK(gradB[i] == doctest::Approx(0.401324f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(-0.266667f));
					CHECK(gradB[i] == doctest::Approx(0.231049f));
				}
			}
		}

		SUBCASE("CrossEntropy Gradient Operation (MeanAll Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 0.3f;
			A[3] = A[4] = A[5] = 0.5f;
			B.Fill(0.4f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = CrossEntropy(A, B, Reduction::Mean, allocator);
			CHECK(C.GetShape() == Shape(1));
			CHECK(C.RequiresGrad());

			C.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			size_t size = gradA.NumElements(); /// The sizes of the gradients of A and B are the same

			for (size_t i = 0; i < size; ++i) {
				if (i < 3) {
					CHECK(gradA[i] == doctest::Approx(-0.222222f));
					CHECK(gradB[i] == doctest::Approx(0.200662f));
				}
				else {
					CHECK(gradA[i] == doctest::Approx(-0.133333f));
					CHECK(gradB[i] == doctest::Approx(0.115525f));
				}
			}
		}

		SUBCASE("Empty tensor CrossEntropy throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			Tensor<float> B(Shape(), allocator);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			CHECK_THROWS_AS(CrossEntropy(A, B, Reduction::None, allocator), std::runtime_error);
		}
	}

	TEST_CASE("CrossEntropyWithLogits Gradient") {
		SUBCASE("CrossEntropyWithLogits Gradient Operation (No Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3.0f;
			A[3] = A[4] = A[5] = 5.0f;
			B.Fill(4.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = CrossEntropyWithLogits(A, B, Reduction::None, allocator);
			CHECK(C.GetShape() == Shape(2, 1));
			CHECK(C.RequiresGrad());

			auto loss = SumAll(C, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 0.0f);
			}

			for (auto& val : gradB) {
				CHECK(val == doctest::Approx(0.366204f));
			}
		}

		SUBCASE("CrossEntropyWithLogits Gradient Operation (SumAll Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3.0f;
			A[3] = A[4] = A[5] = 5.0f;
			B.Fill(4.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = CrossEntropyWithLogits(A, B, Reduction::Sum, allocator);
			CHECK(C.GetShape() == Shape(1));
			CHECK(C.RequiresGrad());

			C.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 0.0f);
			}

			for (auto& val : gradB) {
				CHECK(val == doctest::Approx(0.366204f));
			}
		}

		SUBCASE("CrossEntropyWithLogits Gradient Operation (MeanAll Reduction)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3.0f;
			A[3] = A[4] = A[5] = 5.0f;
			B.Fill(4.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = CrossEntropyWithLogits(A, B, Reduction::Mean, allocator);
			CHECK(C.GetShape() == Shape(1));
			CHECK(C.RequiresGrad());

			C.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 0.0f);
			}

			for (auto& val : gradB) {
				CHECK(val == doctest::Approx(0.183102f));
			}
		}

		SUBCASE("Empty tensor CrossEntropyWithLogits throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			Tensor<float> B(Shape(), allocator);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			CHECK_THROWS_AS(CrossEntropyWithLogits(A, B, Reduction::None, allocator), std::runtime_error);
		}
	}
}