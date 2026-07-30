/// scalarTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/scalar/scalar.h>

using namespace MLCore::Utils;

using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Scalar Operations Tests") {
    TEST_CASE("AddScalar") {
        SUBCASE("AddScalar adds scalar to each element") {
            Tensor<float> A({ 4 });

            A[0] = 1;
            A[1] = 2;
            A[2] = 3;
            A[3] = 4;

            auto B = AddScalar(A, 5.0f);

            CHECK(B.GetShape() == Shape(4));

            CHECK(B[0] == doctest::Approx(6));
            CHECK(B[1] == doctest::Approx(7));
            CHECK(B[2] == doctest::Approx(8));
            CHECK(B[3] == doctest::Approx(9));
        }

        SUBCASE("AddScalar propagates requires-grad") {
            Tensor<float> A({ 2 });

            A.SetRequiresGrad(true);

            auto B = AddScalar(A, 3.0f);

            CHECK(B.RequiresGrad());
        }
    }

    TEST_CASE("SubtractScalar") {
        SUBCASE("SubtractScalar computes tensor minus scalar") {
            Tensor<float> A({ 3 });

            A[0] = 5;
            A[1] = 6;
            A[2] = 7;

            auto B = SubtractScalar(A, 2.0f, false);

            CHECK(B[0] == doctest::Approx(3));
            CHECK(B[1] == doctest::Approx(4));
            CHECK(B[2] == doctest::Approx(5));
        }

        SUBCASE("SubtractScalar computes scalar minus tensor") {
            Tensor<float> A({ 3 });

            A[0] = 5;
            A[1] = 6;
            A[2] = 7;

            auto B = SubtractScalar(A, 10.0f, true);

            CHECK(B[0] == doctest::Approx(5));
            CHECK(B[1] == doctest::Approx(4));
            CHECK(B[2] == doctest::Approx(3));
        }

        SUBCASE("SubtractScalar propogates requires-grad") {
            Tensor<float> A({ 2 });

            A.SetRequiresGrad(true);

            CHECK(SubtractScalar(A, 1.0f, false).RequiresGrad());
            CHECK(SubtractScalar(A, 1.0f, true).RequiresGrad());
        }
    }

    TEST_CASE("MultiplyScalar") {
        SUBCASE("MultiplyScalar multiplies each element") {
            Tensor<float> A({ 3 });

            A[0] = 2;
            A[1] = -3;
            A[2] = 4;

            auto B = MultiplyScalar(A, 2.0f);

            CHECK(B[0] == doctest::Approx(4));
            CHECK(B[1] == doctest::Approx(-6));
            CHECK(B[2] == doctest::Approx(8));
        }

        SUBCASE("MultiplyScalar propagates requires-grad") {
            Tensor<float> A({ 2 });

            A.SetRequiresGrad(true);

            auto B = MultiplyScalar(A, 2.0f);

            CHECK(B.RequiresGrad());
        }
    }

    TEST_CASE("DivideScalar") {
        SUBCASE("DivideScalar computes tensor divided by scalar") {
            Tensor<float> A({ 3 });

            A[0] = 6;
            A[1] = 8;
            A[2] = 10;

            auto B = DivideScalar(A, 2.0f, false);

            CHECK(B[0] == doctest::Approx(3));
            CHECK(B[1] == doctest::Approx(4));
            CHECK(B[2] == doctest::Approx(5));
        }

        SUBCASE("DivideScalar computes scalar divided by tensor") {
            Tensor<float> A({ 3 });

            A[0] = 2;
            A[1] = 4;
            A[2] = 5;

            auto B = DivideScalar(A, 20.0f, true);

            CHECK(B[0] == doctest::Approx(10));
            CHECK(B[1] == doctest::Approx(5));
            CHECK(B[2] == doctest::Approx(4));
        }

        SUBCASE("DivideScalar throws when dividing by zero scalar") {
            Tensor<float> A({ 2 });

            A[0] = 1;
            A[1] = 2;

            CHECK_THROWS_AS(
                DivideScalar(A, 0.0f, false),
                std::runtime_error
            );
        }

        SUBCASE("DivideScalar throws when tensor contains zero") {
            Tensor<float> A({ 3 });

            A[0] = 1;
            A[1] = 0;
            A[2] = 2;

            CHECK_THROWS_AS(
                DivideScalar(A, 10.0f, true),
                std::runtime_error
            );
        }

        SUBCASE("DivideScalar propogates requires-grad") {
            Tensor<float> A({ 2 });
            A.Fill(5.0f);

            A.SetRequiresGrad(true);

            auto B = DivideScalar(A, 2.0f, false);
            auto C = DivideScalar(A, 2.0f, true);

            CHECK(B.RequiresGrad());
            CHECK(C.RequiresGrad());
        }

        SUBCASE("DivideScalar integer division behaves correctly") {
            Tensor<int> A({ 2 });

            A[0] = 5;
            A[1] = 7;

            auto B = DivideScalar(A, 2, false);

            CHECK(B[0] == 2);
            CHECK(B[1] == 3);
        }
    }

    TEST_CASE("Scalar operations preserve shape") {
        Tensor<float> A({ 2, 3 });

        CHECK(AddScalar(A, 1.0f).GetShape() == Shape(2, 3));
        CHECK(MultiplyScalar(A, 2.0f).GetShape() == Shape(2, 3));
        CHECK(SubtractScalar(A, 1.0f, false).GetShape() == Shape(2, 3));
        CHECK(DivideScalar(A, 2.0f, false).GetShape() == Shape(2, 3));
    }
}