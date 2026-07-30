/// reductionGradTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/reduction/reduction.h>

using namespace MLCore::Utils;

using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Reduction Gradient Tests") {
	TEST_CASE("SumAll Gradient") {
		SUBCASE("SumAll Gradient Operation") {
			Tensor<float> A({ 2, 3 });
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = 5;
			A.SetRequiresGrad(true);

			auto B = SumAll(A);
			CHECK(B.GetShape() == Shape(1));
			CHECK(B.RequiresGrad());

			B.Backward(); /// No need for loss here, it's already scalar

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			
			for (auto& val : gradA) {
				CHECK(val == 1);
			}
		}

		SUBCASE("Empty tensor SumAll returns 0") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			auto B = SumAll(A);
			CHECK(B.GetShape() == Shape(1));

			for (auto& val : B) {
				CHECK(val == 0.0f);
			}
		}
	}

	TEST_CASE("MeanAll Gradient") {
		SUBCASE("MeanAll Gradient Operation") {
			Tensor<float> A({ 2, 3 });
			A[0] = A[1] = A[2] = 3;
			A[3] = A[4] = A[5] = 5;
			A.SetRequiresGrad(true);

			auto B = MeanAll(A);
			CHECK(B.GetShape() == Shape(1));
			CHECK(B.RequiresGrad());

			B.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == doctest::Approx(0.166667));
			}
		}

		SUBCASE("Empty tensor MeanAll throws") {
			Tensor<float> A(Shape(0));
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(MeanAll(A), std::runtime_error);
		}
	}

	TEST_CASE("MaxAll Gradient") {
		SUBCASE("MaxAll Gradient Operation") {
			Tensor<float> A({ 2, 3 });
			A[0] = 1;
			A[1] = 2;
			A[2] = 3;
			A[3] = 4;
			A[4] = 5;
			A[5] = 6;
			A.SetRequiresGrad(true);

			auto B = MaxAll(A);
			CHECK(B.GetShape() == Shape(1));
			CHECK(B.RequiresGrad());

			B.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (size_t i = 0; i < 4; ++i) {
				CHECK(gradA[i] == 0.0f);
			}

			CHECK(gradA[5] == 1.0f);
		}

		SUBCASE("Empty tensor MaxAll throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(MaxAll(A), std::runtime_error);
		}
	}

	TEST_CASE("MinAll Gradient") {
		SUBCASE("MinAll Gradient Operation") {
			Tensor<float> A({ 2, 3 });
			A[0] = 1;
			A[1] = 2;
			A[2] = 3;
			A[3] = 4;
			A[4] = 5;
			A[5] = 6;
			A.SetRequiresGrad(true);

			auto B = MinAll(A);
			CHECK(B.GetShape() == Shape(1));
			CHECK(B.RequiresGrad());

			B.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (size_t i = 1; i < 5; ++i) {
				CHECK(gradA[i] == 0.0f);
			}

			CHECK(gradA[0] == 1.0f);
		}

		SUBCASE("Empty tensor MinAll throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(MinAll(A), std::runtime_error);
		}
	}

	TEST_CASE("AxisSum Gradient") {
		SUBCASE("AxisSum Gradient Operation (Axis 0, No KeepDims)") {
			Tensor<float> A({ 2, 3 });
			A[0] = 1;
			A[1] = 2;
			A[2] = 3;
			A[3] = 4;
			A[4] = 5;
			A[5] = 6;
			A.SetRequiresGrad(true);

			auto B = AxisSum(A, 0);
			CHECK(B.GetShape() == Shape(3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();
			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("AxisSum Gradient Operation (Axis 1 with KeepDims)") {
			Tensor<float> A({ 2, 3 });
			A[0] = 1;
			A[1] = 2;
			A[2] = 3;
			A[3] = 4;
			A[4] = 5;
			A[5] = 6;
			A.SetRequiresGrad(true);

			auto B = AxisSum(A, 1, true);
			CHECK(B.GetShape() == Shape(2, 1));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(A);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("Empty tensor AxisSum throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(AxisSum(A, 0), std::out_of_range);
		}
	}

	TEST_CASE("AxisMean Gradient") {
		SUBCASE("AxisMean Gradient Operation (Axis 0)") {
			Tensor<float> A({ 2, 3 });
			A[0] = 1;
			A[1] = 2;
			A[2] = 3;
			A[3] = 4;
			A[4] = 5;
			A[5] = 6;
			A.SetRequiresGrad(true);

			auto B = AxisMean(A, 0);
			CHECK(B.GetShape() == Shape(3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(A);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("AxisMean Gradient Operation (Axis 1)") {
			Tensor<float> A({ 2, 3 });
			A[0] = 1;
			A[1] = 2;
			A[2] = 3;
			A[3] = 4;
			A[4] = 5;
			A[5] = 6;
			A.SetRequiresGrad(true);

			auto B = AxisMean(A, 1, true);
			CHECK(B.GetShape() == Shape(2, 1));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
		}

		SUBCASE("Empty tensor AxisMean throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(AxisMean(A, 0), std::out_of_range);
		}
	}

	TEST_CASE("AxisMax Gradient") {
		SUBCASE("AxisMax Gradient Operation (Axis 0)") {
			Tensor<float> A({ 2, 3 });
			A[0] = 1;
			A[1] = 2;
			A[2] = 3;
			A[3] = 4;
			A[4] = 5;
			A[5] = 6;
			A.SetRequiresGrad(true);

			auto B = AxisMax(A, 0);
			CHECK(B.GetShape() == Shape(3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(A);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("AxisMax Gradient Operation (Axis 1)") {
			Tensor<float> A({ 2, 3 });
			A[0] = 1;
			A[1] = 2;
			A[2] = 3;
			A[3] = 4;
			A[4] = 5;
			A[5] = 6;
			A.SetRequiresGrad(true);

			auto B = AxisMax(A, 1, true);
			CHECK(B.GetShape() == Shape(2, 1));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
		}

		SUBCASE("Empty tensor AxisMax throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(AxisMax(A, 0), std::out_of_range);
		}
	}

	TEST_CASE("AxisMin Gradient") {
		SUBCASE("AxisMin Gradient Operation (Axis 0)") {
			Tensor<float> A({ 2, 3 });
			A[0] = 1;
			A[1] = 2;
			A[2] = 3;
			A[3] = 4;
			A[4] = 5;
			A[5] = 6;
			A.SetRequiresGrad(true);

			auto B = AxisMin(A, 0);
			CHECK(B.GetShape() == Shape(3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(A);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("AxisMin Gradient Operation (Axis 1)") {
			Tensor<float> A({ 2, 3 });
			A[0] = 1;
			A[1] = 2;
			A[2] = 3;
			A[3] = 4;
			A[4] = 5;
			A[5] = 6;
			A.SetRequiresGrad(true);

			auto B = AxisMin(A, 1, true);
			CHECK(B.GetShape() == Shape(2, 1));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B);
			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
		}

		SUBCASE("Empty tensor AxisMin throws") {
			Tensor<float> A(Shape{});
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(AxisMin(A, 0), std::out_of_range);
		}
	}
}