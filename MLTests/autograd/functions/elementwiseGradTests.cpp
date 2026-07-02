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

			size_t sizeA = gradA.NumElements();
			size_t sizeB = gradB.NumElements();

			for (size_t i = 0; i < sizeA; ++i) {
				CHECK(gradA[i] == 1);
			}

			for (size_t i = 0; i < sizeB; ++i) {
				CHECK(gradB[i] == 1);
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

			size_t sizeA = gradA.NumElements();
			size_t sizeB = gradB.NumElements();

			for (size_t i = 0; i < sizeA; ++i) {
				CHECK(gradA[i] == 1);
			}

			for (size_t i = 0; i < sizeB; ++i) {
				CHECK(gradB[i] == 3);
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

			size_t sizeA = gradA.NumElements();
			size_t sizeB = gradB.NumElements();

			for (size_t i = 0; i < sizeA; ++i) {
				CHECK(gradA[i] == 1);
			}

			for (size_t i = 0; i < sizeB; ++i) {
				CHECK(gradB[i] == -1);
			}
		}

		SUBCASE("Add Gradient with broadcasting") {

		}
	}

	TEST_CASE("Elementwise Multiply Gradient") {

	}

	TEST_CASE("Elementwise Divide Gradient") {

	}

	TEST_CASE("Elementsise Power Gradient") {

	}

	TEST_CASE("Elementwise Abs Gradient") {

	}

	TEST_CASE("Elementwise Clamp Gradient") {

	}

	TEST_CASE("Elementwise Log Gradient") {

	}

	TEST_CASE("Elementwise Exp Gradient") {

	}

	TEST_CASE("Elementwise Equal Gradient") {

	}
}