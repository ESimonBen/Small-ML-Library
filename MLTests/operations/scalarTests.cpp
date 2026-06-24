/// scalarTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/scalar/scalar.h>

using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Scalar Operations Tests") {
    TEST_CASE("AddScalar") {
        SUBCASE("AddScalar adds scalar to each element") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = 1;
            A[1] = 2;
            A[2] = 3;
            A[3] = 4;

            auto B = AddScalar(A, 5.0f, allocator);

            CHECK(B.GetShape() == Shape(4));

            CHECK(B[0] == doctest::Approx(6));
            CHECK(B[1] == doctest::Approx(7));
            CHECK(B[2] == doctest::Approx(8));
            CHECK(B[3] == doctest::Approx(9));
        }

        SUBCASE("AddScalar propagates requires-grad") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2 }, allocator);

            A.SetRequiresGrad(true);

            auto B = AddScalar(A, 3.0f, allocator);

            CHECK(B.RequiresGrad());
        }
    }

    TEST_CASE("SubtractScalar") {
        SUBCASE("SubtractScalar computes tensor minus scalar") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 3 }, allocator);

            A[0] = 5;
            A[1] = 6;
            A[2] = 7;

            auto B = SubtractScalar(A, 2.0f, allocator, false);

            CHECK(B[0] == doctest::Approx(3));
            CHECK(B[1] == doctest::Approx(4));
            CHECK(B[2] == doctest::Approx(5));
        }

        SUBCASE("SubtractScalar computes scalar minus tensor") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 3 }, allocator);

            A[0] = 5;
            A[1] = 6;
            A[2] = 7;

            auto B = SubtractScalar(A, 10.0f, allocator, true);

            CHECK(B[0] == doctest::Approx(5));
            CHECK(B[1] == doctest::Approx(4));
            CHECK(B[2] == doctest::Approx(3));
        }

        SUBCASE("SubtractScalar propogates requires-grad") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2 }, allocator);

            A.SetRequiresGrad(true);

            CHECK(SubtractScalar(A, 1.0f, allocator, false).RequiresGrad());
            CHECK(SubtractScalar(A, 1.0f, allocator, true).RequiresGrad());
        }
    }

    TEST_CASE("MultiplyScalar") {
        SUBCASE("MultiplyScalar multiplies each element") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 3 }, allocator);

            A[0] = 2;
            A[1] = -3;
            A[2] = 4;

            auto B = MultiplyScalar(A, 2.0f, allocator);

            CHECK(B[0] == doctest::Approx(4));
            CHECK(B[1] == doctest::Approx(-6));
            CHECK(B[2] == doctest::Approx(8));
        }

        SUBCASE("MultiplyScalar propagates requires-grad") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2 }, allocator);

            A.SetRequiresGrad(true);

            auto B = MultiplyScalar(A, 2.0f, allocator);

            CHECK(B.RequiresGrad());
        }
    }

    TEST_CASE("DivideScalar") {
        SUBCASE("DivideScalar computes tensor divided by scalar") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 3 }, allocator);

            A[0] = 6;
            A[1] = 8;
            A[2] = 10;

            auto B = DivideScalar(A, 2.0f, allocator, false);

            CHECK(B[0] == doctest::Approx(3));
            CHECK(B[1] == doctest::Approx(4));
            CHECK(B[2] == doctest::Approx(5));
        }

        SUBCASE("DivideScalar computes scalar divided by tensor") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 3 }, allocator);

            A[0] = 2;
            A[1] = 4;
            A[2] = 5;

            auto B = DivideScalar(A, 20.0f, allocator, true);

            CHECK(B[0] == doctest::Approx(10));
            CHECK(B[1] == doctest::Approx(5));
            CHECK(B[2] == doctest::Approx(4));
        }

        SUBCASE("DivideScalar throws when dividing by zero scalar") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2 }, allocator);

            A[0] = 1;
            A[1] = 2;

            CHECK_THROWS_AS(
                DivideScalar(A, 0.0f, allocator, false),
                std::runtime_error
            );
        }

        SUBCASE("DivideScalar throws when tensor contains zero") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 3 }, allocator);

            A[0] = 1;
            A[1] = 0;
            A[2] = 2;

            CHECK_THROWS_AS(
                DivideScalar(A, 10.0f, allocator, true),
                std::runtime_error
            );
        }

        SUBCASE("DivideScalar propogates requires-grad") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2 }, allocator);

            A.SetRequiresGrad(true);

            CHECK(DivideScalar(A, 2.0f, allocator, false).RequiresGrad());
            CHECK(DivideScalar(A, 2.0f, allocator, true).RequiresGrad());
        }

        SUBCASE("DivideScalar integer division behaves correctly") {
            ArenaAllocator allocator(1024);

            Tensor<int> A({ 2 }, allocator);

            A[0] = 5;
            A[1] = 7;

            auto B = DivideScalar(A, 2, allocator, false);

            CHECK(B[0] == 2);
            CHECK(B[1] == 3);
        }
    }

    TEST_CASE("Scalar operations preserve shape") {
        ArenaAllocator allocator(1024);

        Tensor<float> A({ 2, 3 }, allocator);

        CHECK(AddScalar(A, 1.0f, allocator).GetShape() == Shape(2, 3));
        CHECK(MultiplyScalar(A, 2.0f, allocator).GetShape() == Shape(2, 3));
        CHECK(SubtractScalar(A, 1.0f, allocator, false).GetShape() == Shape(2, 3));
        CHECK(DivideScalar(A, 2.0f, allocator, false).GetShape() == Shape(2, 3));
    }
}