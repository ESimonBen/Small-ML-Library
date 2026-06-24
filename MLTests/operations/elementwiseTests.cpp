/// elementwiseTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/elementwise/elementwise.h>

using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::Operations;
using namespace MLCore::TensorCore;

TEST_SUITE("Elementwise Operations") {
    TEST_CASE("Elementwise Add") {
        SUBCASE("Elementwise Add: same shapes") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2, 2 }, allocator);
            Tensor<float> B({ 2, 2 }, allocator);

            A[0] = 1; A[1] = 2;
            A[2] = 3; A[3] = 4;

            B[0] = 5; B[1] = 6;
            B[2] = 7; B[3] = 8;

            auto C = Add(A, B, allocator);

            CHECK(C.GetShape() == Shape(2, 2));

            CHECK(C[0] == doctest::Approx(6));
            CHECK(C[1] == doctest::Approx(8));
            CHECK(C[2] == doctest::Approx(10));
            CHECK(C[3] == doctest::Approx(12));
        }

        SUBCASE("Elementwise Add: broadcasting") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2, 3 }, allocator);
            Tensor<float> B({ 3 }, allocator);

            for (size_t i = 0; i < 6; ++i)
                A[i] = static_cast<float>(i + 1);

            B[0] = 10;
            B[1] = 20;
            B[2] = 30;

            auto C = Add(A, B, allocator);

            CHECK(C.GetShape() == Shape(2, 3));

            CHECK(C[0] == doctest::Approx(11));
            CHECK(C[1] == doctest::Approx(22));
            CHECK(C[2] == doctest::Approx(33));
            CHECK(C[3] == doctest::Approx(14));
            CHECK(C[4] == doctest::Approx(25));
            CHECK(C[5] == doctest::Approx(36));
        }

        SUBCASE("Elementwise Add: incompatible shapes throw") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2, 3 }, allocator);
            Tensor<float> B({ 2, 2 }, allocator);

            CHECK_THROWS_AS(
                Add(A, B, allocator),
                std::runtime_error
            );
        }

        SUBCASE("Elementwise Add propagates requires-grad") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2 }, allocator);
            Tensor<float> B({ 2 }, allocator);

            A.SetRequiresGrad(true);

            auto C = Add(A, B, allocator);

            CHECK(C.RequiresGrad());
        }
   }

    TEST_CASE("Elementwise Subtract") {
        SUBCASE("Elementwise Subtract: same shapes") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2, 2 }, allocator);
            Tensor<float> B({ 2, 2 }, allocator);

            A[0] = 1; A[1] = 2;
            A[2] = 3; A[3] = 4;

            B[0] = 5; B[1] = 6;
            B[2] = 7; B[3] = 8;

            auto C = Subtract(B, A, allocator);

            CHECK(C.GetShape() == Shape(2, 2));

            CHECK(C[0] == doctest::Approx(4));
            CHECK(C[1] == doctest::Approx(4));
            CHECK(C[2] == doctest::Approx(4));
            CHECK(C[3] == doctest::Approx(4));
        }

        SUBCASE("Elementwise Subtract: broadcasting") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2, 3 }, allocator);
            Tensor<float> B({ 3 }, allocator);

            for (size_t i = 0; i < 6; ++i)
                A[i] = static_cast<float>(i + 1);

            B[0] = 10;
            B[1] = 20;
            B[2] = 30;

            auto C = Subtract(B, A, allocator);

            CHECK(C.GetShape() == Shape(2, 3));

            CHECK(C[0] == doctest::Approx(9));
            CHECK(C[1] == doctest::Approx(18));
            CHECK(C[2] == doctest::Approx(27));
            CHECK(C[3] == doctest::Approx(6));
            CHECK(C[4] == doctest::Approx(15));
            CHECK(C[5] == doctest::Approx(24));
        }

        SUBCASE("Elementwise Subtract: incompatible shapes throw") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2, 3 }, allocator);
            Tensor<float> B({ 2, 2 }, allocator);

            A.Fill(5.0f);
            B.Fill(3.0f);

            CHECK_THROWS_AS(
                Subtract(A, B, allocator),
                std::runtime_error
            );
        }

        SUBCASE("Elementwise Subtract propagates requires-grad") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2 }, allocator);
            Tensor<float> B({ 2 }, allocator);

            A.Fill(5.0f);
            B.Fill(3.0f);

            A.SetRequiresGrad(true);

            auto C = Subtract(A, B, allocator);

            CHECK(C.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Multiply") {
        SUBCASE("Elementwise Multiply: same shapes") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2, 2 }, allocator);
            Tensor<float> B({ 2, 2 }, allocator);

            A[0] = 1; A[1] = 2;
            A[2] = 3; A[3] = 4;

            B[0] = 5; B[1] = 6;
            B[2] = 7; B[3] = 8;

            auto C = Multiply(A, B, allocator);

            CHECK(C.GetShape() == Shape(2, 2));

            CHECK(C[0] == doctest::Approx(5));
            CHECK(C[1] == doctest::Approx(12));
            CHECK(C[2] == doctest::Approx(21));
            CHECK(C[3] == doctest::Approx(32));
        }

        SUBCASE("Elementwise Multiply: broadcasting") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2, 3 }, allocator);
            Tensor<float> B({ 3 }, allocator);

            for (size_t i = 0; i < 6; ++i)
                A[i] = static_cast<float>(i + 1);

            B[0] = 10;
            B[1] = 20;
            B[2] = 30;

            auto C = Multiply(A, B, allocator);

            CHECK(C.GetShape() == Shape(2, 3));

            CHECK(C[0] == doctest::Approx(10));
            CHECK(C[1] == doctest::Approx(40));
            CHECK(C[2] == doctest::Approx(90));
            CHECK(C[3] == doctest::Approx(40));
            CHECK(C[4] == doctest::Approx(100));
            CHECK(C[5] == doctest::Approx(180));
        }

        SUBCASE("Elementwise Multiply: incompatible shapes throw") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2, 3 }, allocator);
            Tensor<float> B({ 2, 2 }, allocator);

            A.Fill(5.0f);
            B.Fill(3.0f);

            CHECK_THROWS_AS(
                Multiply(A, B, allocator),
                std::runtime_error
            );
        }

        SUBCASE("Elementwise Multiply propagates requires-grad") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2 }, allocator);
            Tensor<float> B({ 2 }, allocator);

            A.Fill(5.0f);
            B.Fill(3.0f);

            A.SetRequiresGrad(true);

            auto C = Multiply(A, B, allocator);

            CHECK(C.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Divide") {
        SUBCASE("Elementwise Divide: same shapes") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2, 2 }, allocator);
            Tensor<float> B({ 2, 2 }, allocator);

            A[0] = 1; A[1] = 2;
            A[2] = 3; A[3] = 4;

            B[0] = 2; B[1] = 4;
            B[2] = 10; B[3] = 10;

            auto C = Divide(A, B, allocator);

            CHECK(C.GetShape() == Shape(2, 2));

            CHECK(C[0] == doctest::Approx(0.5));
            CHECK(C[1] == doctest::Approx(0.5));
            CHECK(C[2] == doctest::Approx(.3));
            CHECK(C[3] == doctest::Approx(.4));
        }

        SUBCASE("Elementwise Divide: broadcasting") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2, 3 }, allocator);
            Tensor<float> B({ 3 }, allocator);

            for (size_t i = 0; i < 6; ++i)
                A[i] = static_cast<float>(i + 1);

            B[0] = 10;
            B[1] = 20;
            B[2] = 30;

            auto C = Divide(B, A, allocator);

            CHECK(C.GetShape() == Shape(2, 3));

            CHECK(C[0] == doctest::Approx(10));
            CHECK(C[1] == doctest::Approx(10));
            CHECK(C[2] == doctest::Approx(10));
            CHECK(C[3] == doctest::Approx(2.5));
            CHECK(C[4] == doctest::Approx(4));
            CHECK(C[5] == doctest::Approx(5));
        }

        SUBCASE("Elementwise Divide: incompatible shapes throw") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2, 3 }, allocator);
            Tensor<float> B({ 2, 2 }, allocator);

            A.Fill(10.0f);
            B.Fill(5.0f);

            CHECK_THROWS_AS(
                Divide(A, B, allocator),
                std::runtime_error
            );
        }

        SUBCASE("Elementwise Divide propagates requires-grad") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 2 }, allocator);
            Tensor<float> B({ 2 }, allocator);

            A.Fill(10.0f);
            B.Fill(5.0f);

            A.SetRequiresGrad(true);

            auto C = Divide(A, B, allocator);

            CHECK(C.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Power") {
        SUBCASE("Elementwise Power Calculation") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            auto B = Power(A, 2.0f, allocator);

            CHECK(B[0] == doctest::Approx(9));
            CHECK(B[1] == doctest::Approx(1));
            CHECK(B[2] == doctest::Approx(0));
            CHECK(B[3] == doctest::Approx(4));
        }

        SUBCASE("Elementwise Power shape preservation") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            auto B = Power(A, 2.0f, allocator);

            CHECK_EQ(B.GetShape(), Shape(4));
        }

        SUBCASE("Elementwise Power propogates requires-grad") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            A.SetRequiresGrad(true);

            auto B = Power(A, 2.0f, allocator);

            CHECK(B.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Abs") {
        SUBCASE("Elementwise Abs Calculation") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            auto B = Abs(A, allocator);

            CHECK(B[0] == doctest::Approx(3));
            CHECK(B[1] == doctest::Approx(1));
            CHECK(B[2] == doctest::Approx(0));
            CHECK(B[3] == doctest::Approx(2));
        }

        SUBCASE("Elementwise Abs shape preservation") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            auto B = Abs(A, allocator);

            CHECK(B.GetShape() == Shape(4));
        }

        SUBCASE("Elementwise Abs propogates requires-grads") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            A.SetRequiresGrad(true);

            auto B = Abs(A, allocator);

            CHECK(B.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Clamp") {
        SUBCASE("Elementwise Clamp Calculation") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            auto B = Clamp(A, -1.0f, 1.0f, allocator);

            CHECK(B[0] == doctest::Approx(-1));
            CHECK(B[1] == doctest::Approx(-1));
            CHECK(B[2] == doctest::Approx(0));
            CHECK(B[3] == doctest::Approx(1));
        }

        SUBCASE("Elementwise Clamp shape preservation") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            auto B = Clamp(A, -1.0f, 1.0f, allocator);

            CHECK(B.GetShape() == Shape(4));
        }

        SUBCASE("Elementwise Clamp propogates requires-grads") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            A.SetRequiresGrad(true);

            auto B = Clamp(A, -1.0f, 1.0f, allocator);

            CHECK(B.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Log") {
        SUBCASE("Elementwise Log Calculation") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = 3;
            A[1] = 10;
            A[2] = 5;
            A[3] = 2;

            auto B = Log(A, allocator);

            CHECK(B[0] == doctest::Approx(1.09861228867));
            CHECK(B[1] == doctest::Approx(2.30258509299));
            CHECK(B[2] == doctest::Approx(1.60943791243));
            CHECK(B[3] == doctest::Approx(0.69314718056));
        }

        SUBCASE("Elementwise Log shape preservation") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = 3;
            A[1] = 1;
            A[2] = 5;
            A[3] = 2;

            auto B = Log(A, allocator);

            CHECK(B.GetShape() == Shape(4));
        }

        SUBCASE("Elementwise Log propogates requires-grads") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = 3;
            A[1] = 1;
            A[2] = 5;
            A[3] = 2;

            A.SetRequiresGrad(true);

            auto B = Log(A, allocator);

            CHECK(B.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Exp") {
        SUBCASE("Elementwise Exp Calculation") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = 3;
            A[1] = 10;
            A[2] = 5;
            A[3] = 2;

            auto B = Exp(A, allocator);

            CHECK(B[0] == doctest::Approx(20.0855369232));
            CHECK(B[1] == doctest::Approx(22026.4657948));
            CHECK(B[2] == doctest::Approx(148.413159103));
            CHECK(B[3] == doctest::Approx(7.38905609893));
        }

        SUBCASE("Elementwise Exp shape preservation") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = 3;
            A[1] = 1;
            A[2] = 5;
            A[3] = 2;

            auto B = Exp(A, allocator);

            CHECK(B.GetShape() == Shape(4));
        }

        SUBCASE("Elementwise Exp propogates requires-grads") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);

            A[0] = 3;
            A[1] = 1;
            A[2] = 5;
            A[3] = 2;

            A.SetRequiresGrad(true);

            auto B = Exp(A, allocator);

            CHECK(B.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Equal") {
        SUBCASE("Elementwise Equal Calculation") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);
            Tensor<float> B({ 4 }, allocator);

            A[0] = 3;
            A[1] = 10;
            A[2] = 5;
            A[3] = 2;

            B[0] = 3;
            B[1] = 10;
            B[2] = 5;
            B[3] = 4;

            auto C = Equal(A, B, allocator);

            CHECK(C[0] == 1);
            CHECK(C[1] == 1);
            CHECK(C[2] == 1);
            CHECK(C[3] == 0);
        }

        SUBCASE("Elementwise Equal with invalid shapes") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);
            Tensor<float> B({ 5 }, allocator);

            A[0] = 3;
            A[1] = 10;
            A[2] = 5;
            A[3] = 2;

            B[0] = 3;
            B[1] = 10;
            B[2] = 5;
            B[3] = 4;
            B[4] = 1;

            CHECK_THROWS_AS(Equal(A, B, allocator), std::runtime_error);
        }

        SUBCASE("Elementwise Equal has no requires-grad") {
            ArenaAllocator allocator(1024);

            Tensor<float> A({ 4 }, allocator);
            Tensor<float> B({ 4 }, allocator);

            A[0] = 3;
            A[1] = 10;
            A[2] = 5;
            A[3] = 2;

            B[0] = 3;
            B[1] = 10;
            B[2] = 5;
            B[3] = 4;

            A.SetRequiresGrad(true);

            auto C = Equal(A, B, allocator);

            CHECK_FALSE(C.RequiresGrad());
        }
    }
}