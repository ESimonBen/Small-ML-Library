/// linAlgTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/linearAlgebra/linalg.h>

using namespace MLCore::Utils;

using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Linear Algebra Operation Tests") {
	TEST_CASE("Matrix Multiply") {
		SUBCASE("MatMultiply Calculation") {
			Tensor<float> A({ 2, 3 });
			Tensor<float> B({ 3, 2 });

			float aVals[] = { 1,2,3,4,5,6 };
			float bVals[] = { 7,8,9,10,11,12 };

			for (size_t i = 0; i < 6; i++) {
				A[i] = aVals[i];
				B[i] = bVals[i];
			}

			auto C = MatMultiply(A, B);

			CHECK(C.GetShape() == Shape({ 2,2 }));

			CHECK(C[0] == doctest::Approx(58.0f));
			CHECK(C[1] == doctest::Approx(64.0f));
			CHECK(C[2] == doctest::Approx(139.0f));
			CHECK(C[3] == doctest::Approx(154.0f));
		}

		SUBCASE("MatMultiply throws on non-2D tensors") {
			Tensor<float> A({ 3 });
			Tensor<float> B({ 3 });

			CHECK_THROWS_AS(MatMultiply(A, B), std::runtime_error);
		}

		SUBCASE("MatMultiply throws on incompatible dimensions") {
			Tensor<float> A({ 3, 2 });
			Tensor<float> B({ 4, 5 });

			CHECK_THROWS_AS(MatMultiply(A, B), std::runtime_error);
		}

		SUBCASE("MatMultiply propagates requires-grad") {
			Tensor<float> A({ 2, 2 });
			Tensor<float> B({ 2, 2 });
			A.Fill(4.0f);
			B.Fill(1.0f);

			A.SetRequiresGrad(true);

			auto C = MatMultiply(A, B);

			CHECK(C.RequiresGrad());
		}

		SUBCASE("MatMultiply with Identity Matrix") {
			Tensor<float> A({ 2, 2 });
			Tensor<float> B({ 2, 2 });
			A.Fill(4.0f);
			
			B[0] = 1;
			B[1] = 0;
			B[2] = 0;
			B[3] = 1;

			auto C = MatMultiply(A, B);

			CHECK(C.GetShape() == A.GetShape());
			CHECK(C[0] == A[0]);
			CHECK(C[1] == A[1]);
			CHECK(C[2] == A[2]);
			CHECK(C[3] == A[3]);
		}
	}

	TEST_CASE("Transpose") {
		SUBCASE("Transpose Operation") {
			Tensor<float> A({ 2, 3 });
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			auto B = Transpose(A);

			CHECK(B.GetShape() == Shape({ 3,2 }));
			CHECK(B[0] == 1);
			CHECK(B[1] == 4);
			CHECK(B[2] == 2);
			CHECK(B[3] == 5);
			CHECK(B[4] == 3);
			CHECK(B[5] == 6);
		}

		SUBCASE("Transpose throws on non-2D tensors") {
			Tensor<float> A({ 3 });

			CHECK_THROWS_AS(Transpose(A), std::runtime_error);
		}

		SUBCASE("Transpose propagates requires-grad") {
			Tensor<float> A({ 2,2 });

			A.Fill(1.0f);
			A.SetRequiresGrad(true);

			auto B = Transpose(A);

			CHECK(B.RequiresGrad());
		}

		SUBCASE("Double transpose returns the original tensor") {
			Tensor<float> A({ 2, 3 });
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			auto B = Transpose(Transpose(A));

			CHECK(B.GetShape() == A.GetShape());
			CHECK(B[0] == A[0]);
			CHECK(B[1] == A[1]);
			CHECK(B[2] == A[2]);
			CHECK(B[3] == A[3]);
			CHECK(B[4] == A[4]);
			CHECK(B[5] == A[5]);
		}
	}

	TEST_CASE("Dot Product") {
		SUBCASE("Dot Product Calculation") {
			Tensor<float> A({ 3 });
			Tensor<float> B({ 3 });

			A[0] = 1;
			A[1] = 2;
			A[2] = 3;

			B[0] = 4;
			B[1] = 5;
			B[2] = 6;

			auto C = Dot(A, B);

			CHECK(C.GetShape() == Shape(1));
			CHECK(C[0] == 32);
		}

		SUBCASE("Dot throws on non-1D tensors") {
			Tensor<float> A({ 2,2 });
			Tensor<float> B({ 2,2 });

			CHECK_THROWS_AS(Dot(A, B), std::runtime_error);
		}

		SUBCASE("Dot throws on unequal lengths") {
			Tensor<float> A({ 3 });
			Tensor<float> B({ 4 });

			CHECK_THROWS_AS(Dot(A, B), std::runtime_error);
		}

		SUBCASE("Dot propagates requires-grad") {
			Tensor<float> A({ 3 });
			Tensor<float> B({ 3 });

			A.Fill(1.0f);
			B.Fill(2.0f);

			B.SetRequiresGrad(true);

			auto C = Dot(A, B);

			CHECK(C.RequiresGrad());
		}

		SUBCASE("Dot Orthogonality") {
			Tensor<float> A({ 2 });
			Tensor<float> B({ 2 });

			A[0] = 1;
			A[1] = 0;
			B[0] = 0;
			B[1] = 1;

			auto C = Dot(A, B);

			CHECK(C[0] == 0);
		}
	}
}