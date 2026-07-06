/// elementwiseGradTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/elementwise/elementwise.h>

using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Elementwise Gradient Tests") {
	TEST_CASE("Elementwise Add Gradient") {
		SUBCASE("Add Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A.Fill(4.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Add(A, B, allocator);
			CHECK(C.GetShape() == Shape(2, 3));

			auto loss = SumAll(C, allocator);
			loss.Backward();
			
			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 1);
			}

			for (auto& val : gradB) {
				CHECK(val == 1);
			}
		}

		SUBCASE("Add Gradient with broadcasting") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 1 }, allocator);
			A.Fill(4.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Add(A, B, allocator);

			CHECK(C.GetShape() == Shape(2, 3));

			auto loss = SumAll(C, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 1));

			for (auto& val : gradA) {
				CHECK(val == 1);
			}

			for (auto& val : gradB) {
				CHECK(val == 3);
			}
		}
	}

	TEST_CASE("Elementwise Subtract Gradient") {
		SUBCASE("Subtract Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A.Fill(7.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Subtract(A, B, allocator);
			CHECK(C.GetShape() == Shape(2, 3));

			auto loss = SumAll(C, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 1);
			}

			for (auto& val : gradB) {
				CHECK(val == -1);
			}
		}
		SUBCASE("Subtract Gradient with broadcasting") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 1 }, allocator);
			A.Fill(4.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Subtract(A, B, allocator);

			CHECK(C.GetShape() == Shape(2, 3));

			auto loss = SumAll(C, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 1));

			for (auto& val : gradA) {
				CHECK(val == 1);
			}

			for (auto& val : gradB) {
				CHECK(val == -3);
			}
		}
	}

	TEST_CASE("Elementwise Multiply Gradient") {
		SUBCASE("Multiply Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A.Fill(2.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Multiply(A, B, allocator);
			CHECK(C.GetShape() == Shape(2, 3));

			auto loss = SumAll(C, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 3);
			}

			for (auto& val : gradB) {
				CHECK(val == 2);
			}
		}

		SUBCASE("Multiply Gradient with broadcasting") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 1 }, allocator);
			A.Fill(2.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Multiply(A, B, allocator);

			CHECK(C.GetShape() == Shape(2, 3));

			auto loss = SumAll(C, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 1));

			for (auto& val : gradA) {
				CHECK(val == 3);
			}

			for (auto& val : gradB) {
				CHECK(val == 6);
			}
		}
	}

	TEST_CASE("Elementwise Divide Gradient") {
		SUBCASE("Divide Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A.Fill(2.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Divide(A, B, allocator);
			CHECK(C.GetShape() == Shape(2, 3));

			auto loss = SumAll(C, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == doctest::Approx(1.0f / 3.0f));
			}

			for (auto& val : gradB) {
				CHECK(val == doctest::Approx(-2.0f / 9.0f));
			}
		}

		SUBCASE("Divide Gradient with broadcasting") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 1 }, allocator);
			A.Fill(2.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Divide(A, B, allocator);

			CHECK(C.GetShape() == Shape(2, 3));

			auto loss = SumAll(C, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();


			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(2, 1));

			for (auto& val : gradA) {
				CHECK(val == doctest::Approx(1.0f / 3.0f));
			}

			for (auto& val : gradB) {
				CHECK(val == doctest::Approx(3.0f * (-2.0f / 9.0f)));
			}
		}
	}

	TEST_CASE("Elementwise Power Gradient") {
		SUBCASE("Power Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(3.0f);
			A.SetRequiresGrad(true);

			auto B = Power(A, 2.0f, allocator);
			
			CHECK(B.GetShape() == Shape(2, 3));
			
			auto loss = SumAll(B, allocator);
			loss.Backward();
			
			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 6.0f);
			}
		}

		SUBCASE("Power Gradient Operation (Fraction Power)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(4.0f);
			A.SetRequiresGrad(true);

			auto B = Power(A, 0.5f, allocator);

			CHECK(B.GetShape() == Shape(2, 3));

			auto loss = SumAll(B, allocator);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == doctest::Approx(0.25f));
			}
		}

		SUBCASE("Power Gradient Operation (Zero Power)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(4.0f);
			A.SetRequiresGrad(true);

			auto B = Power(A, 0.0f, allocator);

			CHECK(B.GetShape() == Shape(2, 3));

			auto loss = SumAll(B, allocator);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 0.0f);
			}
		}

		SUBCASE("Null input throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Power(A, 0.0f, allocator), std::runtime_error);
		}
	}

	TEST_CASE("Elementwise Abs Gradient") {
		SUBCASE("Abs Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(-4.0f);
			A.SetRequiresGrad(true);

			auto B = Abs(A, allocator);
			CHECK(B.GetShape() == Shape(2, 3));

			auto loss = SumAll(B, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == -1.0f);
			}
		}

		SUBCASE("Null input throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Abs(A, allocator), std::runtime_error);
		}
	}

	TEST_CASE("Elementwise Clamp Gradient") {
		SUBCASE("Clamp Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(0.3f);
			A.SetRequiresGrad(true);

			auto B = Clamp(A, -0.5f, 0.5f, allocator);
			CHECK(B.GetShape() == Shape(2, 3));

			auto loss = SumAll(B, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("Null input throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Clamp(A, -0.5f, 0.5f, allocator), std::runtime_error);
		}
	}

	TEST_CASE("Elementwise Log Gradient") {
		SUBCASE("Log Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(2.0f);
			A.SetRequiresGrad(true);

			auto B = Log(A, allocator);
			CHECK(B.GetShape() == Shape(2, 3));

			auto loss = SumAll(B, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 0.5f);
			}
		}

		SUBCASE("Null input throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Log(A, allocator), std::runtime_error);
		}
	}

	TEST_CASE("Elementwise Exp Gradient") {
		SUBCASE("Exp Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(2.0f);
			A.SetRequiresGrad(true);

			auto B = Exp(A, allocator);
			CHECK(B.GetShape() == Shape(2, 3));

			auto loss = SumAll(B, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == doctest::Approx(7.389));
			}
		}

		SUBCASE("Null input throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Exp(A, allocator), std::runtime_error);
		}
	}

	TEST_CASE("Elementwise Equal Gradient") {
		SUBCASE("Equal Gradient Operation (Equal)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A.Fill(2.0f);
			B.Fill(2.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Equal(A, B, allocator);
			CHECK(C.GetShape() == Shape(2, 3));

			auto loss = SumAll(B, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 0.0f);
			}
		}

		SUBCASE("Equal Gradient Operation (Not Equal)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);
			A.Fill(2.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Equal(A, B, allocator);
			CHECK(C.GetShape() == Shape(2, 3));

			auto loss = SumAll(B, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 0.0f);
			}
		}

		SUBCASE("Shape mismatch throws") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 1 }, allocator);
			A.Fill(3.0f);
			B.Fill(4.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			CHECK_THROWS_AS(Equal(A, B, allocator), std::runtime_error);
		}

		SUBCASE("Null input throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			Tensor<float> B(Shape(), allocator);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			CHECK_THROWS_AS(Equal(A, B, allocator), std::runtime_error);
		}
	}
}