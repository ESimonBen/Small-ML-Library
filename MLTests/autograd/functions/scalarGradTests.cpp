/// scalarGradTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/scalar/scalar.h>

using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Scalar Gradient Tests") {
	TEST_CASE("Scalar Add Gradient") {
		SUBCASE("Scalar Add Gradient Operation") {
			ArenaAllocator allocator;
			
			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(3.0f);
			A.SetRequiresGrad(true);

			auto B = AddScalar(A, 4.0f, allocator);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

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

			CHECK_THROWS_AS(AddScalar(A, 2.0f, allocator), std::runtime_error);
		}
	}
	
	TEST_CASE("Scalar Subtract Gradient") {
		SUBCASE("Scalar Subtract Gradient Operation (Scalar on Right)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(3.0f);
			A.SetRequiresGrad(true);

			auto B = SubtractScalar(A, 4.0f, allocator, false);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("Scalar Subtract Gradient Operation (Scalar on Left)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(3.0f);
			A.SetRequiresGrad(true);

			auto B = SubtractScalar(A, 4.0f, allocator, true);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

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

			CHECK_THROWS_AS(SubtractScalar(A, 2.0f, allocator, false), std::runtime_error);
		}
	}

	TEST_CASE("Scalar Multiply Gradient") {
		SUBCASE("Scalar Multiply Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(3.0f);
			A.SetRequiresGrad(true);

			auto B = MultiplyScalar(A, 4.0f, allocator);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 4.0f);
			}
		}

		SUBCASE("Null input throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(MultiplyScalar(A, 2.0f, allocator), std::runtime_error);
		}
	}

	TEST_CASE("Scalar Divide Gradient") {
		SUBCASE("Scalar Divide Gradient Operation (Scalar on Right)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(3.0f);
			A.SetRequiresGrad(true);

			auto B = DivideScalar(A, 4.0f, allocator, false);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == doctest::Approx(0.25f));
			}
		}

		SUBCASE("Scalar Divide Gradient Operation (Scalar on Left)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(3.0f);
			A.SetRequiresGrad(true);

			auto B = DivideScalar(A, 4.0f, allocator, true);
			CHECK(B.GetShape() == Shape(2, 3));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == doctest::Approx(-0.444444f));
			}
		}

		SUBCASE("Null input throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(DivideScalar(A, 2.0f, allocator, false), std::runtime_error);
		}
	}
}