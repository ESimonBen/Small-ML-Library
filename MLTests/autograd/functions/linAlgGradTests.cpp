/// linAlgGradTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/linearAlgebra/linalg.h>

using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Linear Algebra Gradient Tests") {
	TEST_CASE("MatMultiply Gradient") {
		SUBCASE("MatMultiply Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 3, 1 }, allocator);
			
			A[0] = A[1] = A[2] = 3.0f;
			A[3] = A[4] = A[5] = 5.0f;
			B.Fill(4.0f);

			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = MatMultiply(A, B, allocator);
			CHECK(C.GetShape() == Shape(2, 1));
			CHECK(C.RequiresGrad());

			auto loss = SumAll(C, allocator);
			loss.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));
			CHECK(gradB.GetShape() == Shape(3, 1));

			for (auto& val : gradA) {
				CHECK(val == 4.0f);
			}

			for (auto& val : gradB) {
				CHECK(val == 8.0f);
			}
		}

		SUBCASE("MatMultiply throws on dimension mismatch/non-matrix tensors") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			Tensor<float> B({ 2, 3 }, allocator);

			A.Fill(4.0f);
			B.Fill(3.0f);

			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			CHECK_THROWS_AS(MatMultiply(A, B, allocator), std::runtime_error);
		}

		SUBCASE("Empty tensor MatMultiply throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			Tensor<float> B(Shape(), allocator);

			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			CHECK_THROWS_AS(MatMultiply(A, B, allocator), std::runtime_error);
		}
	}

	TEST_CASE("Transpose Gradient") {
		SUBCASE("Transpose Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A[0] = A[1] = A[2] = 3.0f;
			A[3] = A[4] = A[5] = 5.0f;
			
			A.SetRequiresGrad(true);

			auto B = Transpose(A, allocator);
			CHECK(B.GetShape() == Shape(3, 2));
			CHECK(B.RequiresGrad());

			auto loss = SumAll(B, allocator);

			loss.Backward();

			auto gradA = A.Grad();

			CHECK(gradA.GetShape() == Shape(2, 3));

			for (auto& val : gradA) {
				CHECK(val == 1.0f);
			}
		}

		SUBCASE("Transpose throws on non-matrix tensors") {
			ArenaAllocator allocator;

			Tensor<float> A({2, 3, 4}, allocator);
			A.Fill(2.0f);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Transpose(A, allocator), std::runtime_error);
		}

		SUBCASE("Empty tensor Transpose throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			A.SetRequiresGrad(true);

			CHECK_THROWS_AS(Transpose(A, allocator), std::runtime_error);
		}
	}

	TEST_CASE("Dot Product Gradient") {
		SUBCASE("Dot Product Gradient Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 3 }, allocator);
			Tensor<float> B({ 3 }, allocator);
			A.Fill(4.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			auto C = Dot(A, B, allocator);
			CHECK(C.GetShape() == Shape(1));

			C.Backward();

			auto gradA = A.Grad();
			auto gradB = B.Grad();

			CHECK(gradA.GetShape() == Shape(3));
			CHECK(gradB.GetShape() == Shape(3));

			for (auto& val : gradA) {
				CHECK(val == 3.0f);
			}

			for (auto& val : gradB) {
				CHECK(val == 4.0f);
			}
		}

		SUBCASE("Dot product throws on dimension mismatch/non-vector tensors") {
			ArenaAllocator allocator;

			Tensor<float> A({ 3, 2 }, allocator);
			Tensor<float> B({ 3, 1 }, allocator);
			A.Fill(4.0f);
			B.Fill(3.0f);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			CHECK_THROWS_AS(Dot(A, B, allocator), std::runtime_error);
		}

		SUBCASE("Empty tensor Dot throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);
			Tensor<float> B(Shape(), allocator);
			A.SetRequiresGrad(true);
			B.SetRequiresGrad(true);

			CHECK_THROWS_AS(Dot(A, B, allocator), std::runtime_error);
		}
	}
}