/// elementwiseGradTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/elementwise/elementwise.h>

using namespace MLCore::Utils;

using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Elementwise Gradient Tests") {
	TEST_CASE("Elementwise Add Gradient") {
		SUBCASE("Add Gradient Operation") {
			Tensor<float> A({ 2, 3 });
			Tensor<float> B({ 2, 3 });
			A.Fill(4.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Add(A, B);
			CHECK(C.GetShape() == Shape(2, 3));
			CHECK(C.RequiresGrad());

			auto loss = SumAll(C);
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
			Tensor<float> A({ 2, 3 });
			Tensor<float> B({ 2, 1 });
			A.Fill(4.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Add(A, B);
			CHECK(C.GetShape() == Shape(2, 3));
			CHECK(C.RequiresGrad());

			auto loss = SumAll(C);
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

		SUBCASE("Null input throws") {
			Tensor<float> A(Shape{});
			Tensor<float> B(Shape{});
			
			CHECK_THROWS_AS(Add(A, B), std::runtime_error);
		}
	}

	TEST_CASE("Elementwise Subtract Gradient") {
		SUBCASE("Subtract Gradient Operation") {
			Tensor<float> A({ 2, 3 });
			Tensor<float> B({ 2, 3 });
			A.Fill(7.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Subtract(A, B);
			CHECK(C.GetShape() == Shape(2, 3));
			CHECK(C.RequiresGrad());

			auto loss = SumAll(C);
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
			Tensor<float> A({ 2, 3 });
			Tensor<float> B({ 2, 1 });
			A.Fill(4.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Subtract(A, B);
			CHECK(C.GetShape() == Shape(2, 3));
			CHECK(C.RequiresGrad());

			auto loss = SumAll(C);
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
			Tensor<float> A({ 2, 3 });
			Tensor<float> B({ 2, 3 });
			A.Fill(2.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Multiply(A, B);
			CHECK(C.GetShape() == Shape(2, 3));
			CHECK(C.RequiresGrad());

			auto loss = SumAll(C);
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
			Tensor<float> A({ 2, 3 });
			Tensor<float> B({ 2, 1 });
			A.Fill(2.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Multiply(A, B);
			CHECK(C.GetShape() == Shape(2, 3));
			CHECK(C.RequiresGrad());

			auto loss = SumAll(C);
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
			Tensor<float> A({ 2, 3 });
			Tensor<float> B({ 2, 3 });
			A.Fill(2.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Divide(A, B);
			CHECK(C.GetShape() == Shape(2, 3));
			CHECK(C.RequiresGrad());

			auto loss = SumAll(C);
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
			Tensor<float> A({ 2, 3 });
			Tensor<float> B({ 2, 1 });
			A.Fill(2.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Divide(A, B);
			CHECK(C.GetShape() == Shape(2, 3));
			CHECK(C.RequiresGrad());

			auto loss = SumAll(C);
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
			Tensor<float> A({ 2, 3 });
			A.Fill(3.0f);
			A.SetRequiresGrad(true);

			auto B = Power(A, 2.0f);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());
			
			auto loss = SumAll(B);
			loss.Backward();
			
			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 6.0f);
			}
		}

		SUBCASE("Power Gradient Operation (Fraction Power)") {
			Tensor<float> A({ 2, 3 });
			A.Fill(4.0f);
			A.SetRequiresGrad(true);

			auto B = Power(A, 0.5f);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == doctest::Approx(0.25f));
			}
		}

		SUBCASE("Power Gradient Operation (Zero Power)") {
			Tensor<float> A({ 2, 3 });
			A.Fill(4.0f);
			A.SetRequiresGrad(true);

			auto B = Power(A, 0.0f);
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

		SUBCASE("Null input throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Power(A, 0.0f), std::runtime_error);
		}
	}

	TEST_CASE("Elementwise Abs Gradient") {
		SUBCASE("Abs Gradient Operation") {
			Tensor<float> A({ 2, 3 });
			A.Fill(-4.0f);
			A.SetRequiresGrad(true);

			auto B = Abs(A);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();
			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == -1.0f);
			}
		}

		SUBCASE("Null input throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Abs(A), std::runtime_error);
		}
	}

	TEST_CASE("Elementwise Clamp Gradient") {
		SUBCASE("Clamp Gradient Operation") {
			Tensor<float> A({ 2, 3 });
			A.Fill(0.3f);
			A.SetRequiresGrad(true);

			auto B = Clamp(A, -0.5f, 0.5f);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();
			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("Null input throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Clamp(A, -0.5f, 0.5f), std::runtime_error);
		}
	}

	TEST_CASE("Elementwise Log Gradient") {
		SUBCASE("Log Gradient Operation") {
			Tensor<float> A({ 2, 3 });
			A.Fill(2.0f);
			A.SetRequiresGrad(true);

			auto B = Log(A);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();
			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 0.5f);
			}
		}

		SUBCASE("Null input throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Log(A), std::runtime_error);
		}
	}

	TEST_CASE("Elementwise Exp Gradient") {
		SUBCASE("Exp Gradient Operation") {
			Tensor<float> A({ 2, 3 });
			A.Fill(2.0f);
			A.SetRequiresGrad(true);

			auto B = Exp(A);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();
			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == doctest::Approx(7.389));
			}
		}

		SUBCASE("Null input throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Exp(A), std::runtime_error);
		}
	}
}