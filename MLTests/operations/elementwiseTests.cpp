/// elementwiseTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/elementwise/elementwise.h>

using namespace MLCore::Utils;

using namespace MLCore::Operations;
using namespace MLCore::TensorCore;

TEST_SUITE("Elementwise Operations") {
    TEST_CASE("Elementwise Add") {
        SUBCASE("Elementwise Add: same shapes") {
            Tensor<float> A({ 2, 2 });
            Tensor<float> B({ 2, 2 });

            A[0] = 1; A[1] = 2;
            A[2] = 3; A[3] = 4;

            B[0] = 5; B[1] = 6;
            B[2] = 7; B[3] = 8;

            auto C = Add(A, B);

            CHECK(C.GetShape() == Shape(2, 2));

            CHECK(C[0] == doctest::Approx(6));
            CHECK(C[1] == doctest::Approx(8));
            CHECK(C[2] == doctest::Approx(10));
            CHECK(C[3] == doctest::Approx(12));
        }

        SUBCASE("Elementwise Add: broadcasting") {
            Tensor<float> A({ 2, 3 });
            Tensor<float> B({ 3 });

            for (size_t i = 0; i < 6; ++i)
                A[i] = static_cast<float>(i + 1);

            B[0] = 10;
            B[1] = 20;
            B[2] = 30;

            auto C = Add(A, B);

            CHECK(C.GetShape() == Shape(2, 3));

            CHECK(C[0] == doctest::Approx(11));
            CHECK(C[1] == doctest::Approx(22));
            CHECK(C[2] == doctest::Approx(33));
            CHECK(C[3] == doctest::Approx(14));
            CHECK(C[4] == doctest::Approx(25));
            CHECK(C[5] == doctest::Approx(36));
        }

        SUBCASE("Elementwise Add: incompatible shapes throw") {
            Tensor<float> A({ 2, 3 });
            Tensor<float> B({ 2, 2 });

            CHECK_THROWS_AS(
                Add(A, B),
                std::runtime_error
            );
        }

        SUBCASE("Elementwise Add propagates requires-grad") {
            Tensor<float> A({ 2 });
            Tensor<float> B({ 2 });

            A.SetRequiresGrad(true);

            auto C = Add(A, B);

            CHECK(C.RequiresGrad());
        }
   }

    TEST_CASE("Elementwise Subtract") {
        SUBCASE("Elementwise Subtract: same shapes") {
            Tensor<float> A({ 2, 2 });
            Tensor<float> B({ 2, 2 });

            A[0] = 1; A[1] = 2;
            A[2] = 3; A[3] = 4;

            B[0] = 5; B[1] = 6;
            B[2] = 7; B[3] = 8;

            auto C = Subtract(B, A);

            CHECK(C.GetShape() == Shape(2, 2));

            CHECK(C[0] == doctest::Approx(4));
            CHECK(C[1] == doctest::Approx(4));
            CHECK(C[2] == doctest::Approx(4));
            CHECK(C[3] == doctest::Approx(4));
        }

        SUBCASE("Elementwise Subtract: broadcasting") {
            Tensor<float> A({ 2, 3 });
            Tensor<float> B({ 3 });

            for (size_t i = 0; i < 6; ++i)
                A[i] = static_cast<float>(i + 1);

            B[0] = 10;
            B[1] = 20;
            B[2] = 30;

            auto C = Subtract(B, A);

            CHECK(C.GetShape() == Shape(2, 3));

            CHECK(C[0] == doctest::Approx(9));
            CHECK(C[1] == doctest::Approx(18));
            CHECK(C[2] == doctest::Approx(27));
            CHECK(C[3] == doctest::Approx(6));
            CHECK(C[4] == doctest::Approx(15));
            CHECK(C[5] == doctest::Approx(24));
        }

        SUBCASE("Elementwise Subtract: incompatible shapes throw") {
            Tensor<float> A({ 2, 3 });
            Tensor<float> B({ 2, 2 });

            A.Fill(5.0f);
            B.Fill(3.0f);

            CHECK_THROWS_AS(Subtract(A, B), std::runtime_error);
        }

        SUBCASE("Elementwise Subtract propagates requires-grad") {
            Tensor<float> A({ 2 });
            Tensor<float> B({ 2 });

            A.Fill(5.0f);
            B.Fill(3.0f);

            A.SetRequiresGrad(true);

            auto C = Subtract(A, B);

            CHECK(C.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Multiply") {
        SUBCASE("Elementwise Multiply: same shapes") {
            Tensor<float> A({ 2, 2 });
            Tensor<float> B({ 2, 2 });

            A[0] = 1; A[1] = 2;
            A[2] = 3; A[3] = 4;

            B[0] = 5; B[1] = 6;
            B[2] = 7; B[3] = 8;

            auto C = Multiply(A, B);

            CHECK(C.GetShape() == Shape(2, 2));

            CHECK(C[0] == doctest::Approx(5));
            CHECK(C[1] == doctest::Approx(12));
            CHECK(C[2] == doctest::Approx(21));
            CHECK(C[3] == doctest::Approx(32));
        }

        SUBCASE("Elementwise Multiply: broadcasting") {
            Tensor<float> A({ 2, 3 });
            Tensor<float> B({ 3 });

            for (size_t i = 0; i < 6; ++i)
                A[i] = static_cast<float>(i + 1);

            B[0] = 10;
            B[1] = 20;
            B[2] = 30;

            auto C = Multiply(A, B);

            CHECK(C.GetShape() == Shape(2, 3));

            CHECK(C[0] == doctest::Approx(10));
            CHECK(C[1] == doctest::Approx(40));
            CHECK(C[2] == doctest::Approx(90));
            CHECK(C[3] == doctest::Approx(40));
            CHECK(C[4] == doctest::Approx(100));
            CHECK(C[5] == doctest::Approx(180));
        }

        SUBCASE("Elementwise Multiply: incompatible shapes throw") {
            Tensor<float> A({ 2, 3 });
            Tensor<float> B({ 2, 2 });

            A.Fill(5.0f);
            B.Fill(3.0f);

            CHECK_THROWS_AS(
                Multiply(A, B),
                std::runtime_error
            );
        }

        SUBCASE("Elementwise Multiply propagates requires-grad") {
            Tensor<float> A({ 2 });
            Tensor<float> B({ 2 });

            A.Fill(5.0f);
            B.Fill(3.0f);

            A.SetRequiresGrad(true);

            auto C = Multiply(A, B);

            CHECK(C.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Divide") {
        SUBCASE("Elementwise Divide: same shapes") {
            Tensor<float> A({ 2, 2 });
            Tensor<float> B({ 2, 2 });

            A[0] = 1; A[1] = 2;
            A[2] = 3; A[3] = 4;

            B[0] = 2; B[1] = 4;
            B[2] = 10; B[3] = 10;

            auto C = Divide(A, B);

            CHECK(C.GetShape() == Shape(2, 2));

            CHECK(C[0] == doctest::Approx(0.5));
            CHECK(C[1] == doctest::Approx(0.5));
            CHECK(C[2] == doctest::Approx(.3));
            CHECK(C[3] == doctest::Approx(.4));
        }

        SUBCASE("Elementwise Divide: broadcasting") {
            Tensor<float> A({ 2, 3 });
            Tensor<float> B({ 3 });

            for (size_t i = 0; i < 6; ++i)
                A[i] = static_cast<float>(i + 1);

            B[0] = 10;
            B[1] = 20;
            B[2] = 30;

            auto C = Divide(B, A);

            CHECK(C.GetShape() == Shape(2, 3));

            CHECK(C[0] == doctest::Approx(10));
            CHECK(C[1] == doctest::Approx(10));
            CHECK(C[2] == doctest::Approx(10));
            CHECK(C[3] == doctest::Approx(2.5));
            CHECK(C[4] == doctest::Approx(4));
            CHECK(C[5] == doctest::Approx(5));
        }

        SUBCASE("Elementwise Divide: incompatible shapes throw") {
            Tensor<float> A({ 2, 3 });
            Tensor<float> B({ 2, 2 });

            A.Fill(10.0f);
            B.Fill(5.0f);

            CHECK_THROWS_AS(
                Divide(A, B),
                std::runtime_error
            );
        }

        SUBCASE("Elementwise Divide propagates requires-grad") {
            Tensor<float> A({ 2 });
            Tensor<float> B({ 2 });

            A.Fill(10.0f);
            B.Fill(5.0f);

            A.SetRequiresGrad(true);

            auto C = Divide(A, B);

            CHECK(C.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Power") {
        SUBCASE("Elementwise Power Calculation") {
            Tensor<float> A({ 4 });

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            auto B = Power(A, 2.0f);

            CHECK(B[0] == doctest::Approx(9));
            CHECK(B[1] == doctest::Approx(1));
            CHECK(B[2] == doctest::Approx(0));
            CHECK(B[3] == doctest::Approx(4));
        }

        SUBCASE("Elementwise Power shape preservation") {
            Tensor<float> A({ 4 });

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            auto B = Power(A, 2.0f);

            CHECK_EQ(B.GetShape(), Shape(4));
        }

        SUBCASE("Elementwise Power propogates requires-grad") {
            Tensor<float> A({ 4 });

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            A.SetRequiresGrad(true);

            auto B = Power(A, 2.0f);

            CHECK(B.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Abs") {
        SUBCASE("Elementwise Abs Calculation") {
            Tensor<float> A({ 4 });

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            auto B = Abs(A);

            CHECK(B[0] == doctest::Approx(3));
            CHECK(B[1] == doctest::Approx(1));
            CHECK(B[2] == doctest::Approx(0));
            CHECK(B[3] == doctest::Approx(2));
        }

        SUBCASE("Elementwise Abs shape preservation") {
            Tensor<float> A({ 4 });

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            auto B = Abs(A);

            CHECK(B.GetShape() == Shape(4));
        }

        SUBCASE("Elementwise Abs propogates requires-grads") {
            Tensor<float> A({ 4 });

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            A.SetRequiresGrad(true);

            auto B = Abs(A);

            CHECK(B.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Clamp") {
        SUBCASE("Elementwise Clamp Calculation") {
            Tensor<float> A({ 4 });

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            auto B = Clamp(A, -1.0f, 1.0f);

            CHECK(B[0] == doctest::Approx(-1));
            CHECK(B[1] == doctest::Approx(-1));
            CHECK(B[2] == doctest::Approx(0));
            CHECK(B[3] == doctest::Approx(1));
        }

        SUBCASE("Elementwise Clamp shape preservation") {
            Tensor<float> A({ 4 });

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            auto B = Clamp(A, -1.0f, 1.0f);

            CHECK(B.GetShape() == Shape(4));
        }

        SUBCASE("Elementwise Clamp propogates requires-grads") {
            Tensor<float> A({ 4 });

            A[0] = -3;
            A[1] = -1;
            A[2] = 0;
            A[3] = 2;

            A.SetRequiresGrad(true);

            auto B = Clamp(A, -1.0f, 1.0f);

            CHECK(B.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Log") {
        SUBCASE("Elementwise Log Calculation") {
            Tensor<float> A({ 4 });

            A[0] = 3;
            A[1] = 10;
            A[2] = 5;
            A[3] = 2;

            auto B = Log(A);

            CHECK(B[0] == doctest::Approx(1.09861228867));
            CHECK(B[1] == doctest::Approx(2.30258509299));
            CHECK(B[2] == doctest::Approx(1.60943791243));
            CHECK(B[3] == doctest::Approx(0.69314718056));
        }

        SUBCASE("Elementwise Log shape preservation") {
            Tensor<float> A({ 4 });

            A[0] = 3;
            A[1] = 1;
            A[2] = 5;
            A[3] = 2;

            auto B = Log(A);

            CHECK(B.GetShape() == Shape(4));
        }

        SUBCASE("Elementwise Log propogates requires-grads") {
            Tensor<float> A({ 4 });

            A[0] = 3;
            A[1] = 1;
            A[2] = 5;
            A[3] = 2;

            A.SetRequiresGrad(true);

            auto B = Log(A);

            CHECK(B.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Exp") {
        SUBCASE("Elementwise Exp Calculation") {
            Tensor<float> A({ 4 });

            A[0] = 3;
            A[1] = 10;
            A[2] = 5;
            A[3] = 2;

            auto B = Exp(A);

            CHECK(B[0] == doctest::Approx(20.0855369232));
            CHECK(B[1] == doctest::Approx(22026.4657948));
            CHECK(B[2] == doctest::Approx(148.413159103));
            CHECK(B[3] == doctest::Approx(7.38905609893));
        }

        SUBCASE("Elementwise Exp shape preservation") {
            Tensor<float> A({ 4 });

            A[0] = 3;
            A[1] = 1;
            A[2] = 5;
            A[3] = 2;

            auto B = Exp(A);

            CHECK(B.GetShape() == Shape(4));
        }

        SUBCASE("Elementwise Exp propogates requires-grads") {
            Tensor<float> A({ 4 });

            A[0] = 3;
            A[1] = 1;
            A[2] = 5;
            A[3] = 2;

            A.SetRequiresGrad(true);

            auto B = Exp(A);

            CHECK(B.RequiresGrad());
        }
    }

    TEST_CASE("Elementwise Equal") {
        SUBCASE("Elementwise Equal Calculation") {
            Tensor<float> A({ 4 });
            Tensor<float> B({ 4 });

            A[0] = 3;
            A[1] = 10;
            A[2] = 5;
            A[3] = 2;

            B[0] = 3;
            B[1] = 10;
            B[2] = 5;
            B[3] = 4;

            auto C = Equal(A, B);

            CHECK(C[0] == 1);
            CHECK(C[1] == 1);
            CHECK(C[2] == 1);
            CHECK(C[3] == 0);
        }

        SUBCASE("Elementwise Equal with invalid shapes") {
            Tensor<float> A({ 4 });
            Tensor<float> B({ 5 });

            A[0] = 3;
            A[1] = 10;
            A[2] = 5;
            A[3] = 2;

            B[0] = 3;
            B[1] = 10;
            B[2] = 5;
            B[3] = 4;
            B[4] = 1;

            CHECK_THROWS_AS(Equal(A, B), std::runtime_error);
        }

        SUBCASE("Elementwise Equal has no requires-grad") {
            Tensor<float> A({ 4 });
            Tensor<float> B({ 4 });

            A[0] = 3;
            A[1] = 10;
            A[2] = 5;
            A[3] = 2;

            B[0] = 3;
            B[1] = 10;
            B[2] = 5;
            B[3] = 4;

            A.SetRequiresGrad(true);

            auto C = Equal(A, B);

            CHECK_FALSE(C.RequiresGrad());
        }
    }
}