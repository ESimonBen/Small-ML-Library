/// broadcastTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/broadcast/broadcast.h>

using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Broadcast Tests") {
    TEST_CASE("Squeeze Test") {
        SUBCASE("Squeeze removes leading axis") {
            Tensor<float> A({ 1, 3 });
            A.Fill(3.0f);

            auto B = Squeeze(A, 0);
            CHECK(B.GetShape() == Shape(3));

            for (auto& val : B) {
                CHECK(val == 3.0f);
            }
        }

        SUBCASE("Squeeze removes last axis") {
            Tensor <float> A({ 2, 4, 6, 1 });
            A.Fill(5.0f);

            auto B = Squeeze(A, 3);
            CHECK(B.GetShape() == Shape(2, 4, 6));

            for (auto& val : B) {
                CHECK(val == 5.0f);
            }
        }

        SUBCASE("Squeeze removes any axis") {
            Tensor <float> A({ 2, 1, 6, 3 });
            A.Fill(6.0f);

            auto B = Squeeze(A, 1);
            CHECK(B.GetShape() == Shape(2, 6, 3));

            for (auto& val : B) {
                CHECK(val == 6.0f);
            }
        }

        SUBCASE("Squeeze with out-of-bounds axis throws") {
            Tensor<float> A({2, 3});

            CHECK_THROWS_AS(Squeeze(A, 2), std::out_of_range);
        }
	}

	TEST_CASE("Unsqueeze Test") {
        SUBCASE("Unsqueeze adds first axis") {
            Tensor<float> A({ 2, 3 });
            A.Fill(9.0f);

            auto B = Unsqueeze(A, 0);
            CHECK(B.GetShape() == Shape(1, 2, 3));

            for (auto& val : B) {
                CHECK(val == 9.0f);
            }
        }

        SUBCASE("Unsqueeze adds last axis") {
            Tensor<float> A({ 2, 3 });
            A.Fill(9.0f);

            auto B = Unsqueeze(A, 2);
            CHECK(B.GetShape() == Shape(2, 3, 1));

            for (auto& val : B) {
                CHECK(val == 9.0f);
            }
        }

        SUBCASE("Unsqueeze adds any axis") {
            Tensor<float> A({ 2, 3, 5 });
            A.Fill(9.0f);

            auto B = Unsqueeze(A, 1);
            CHECK(B.GetShape() == Shape(2, 1, 3, 5));

            for (auto& val : B) {
                CHECK(val == 9.0f);
            }
        }

        SUBCASE("Unsqueeze with out-of-bounds axis throws") {
            Tensor<float> A({ 2, 3 });

            CHECK_THROWS_AS(Unsqueeze(A, 3), std::out_of_range);
        }
	}

	TEST_CASE("ReduceSumToShape Test") {
        SUBCASE("ReduceSumToShape reduces broadcast dimension") {
            Tensor<float> grad({ 2,3 });

            grad[0] = 1;
            grad[1] = 2;
            grad[2] = 3;
            grad[3] = 4;
            grad[4] = 5;
            grad[5] = 6;

            auto result = ReduceSumToShape(grad, Shape(1, 3));

            CHECK(result.Dims()[0] == 1);
            CHECK(result[0] == doctest::Approx(5));
            CHECK(result[1] == doctest::Approx(7));
            CHECK(result[2] == doctest::Approx(9));
        }

        SUBCASE("ReduceSumToShape removes leading dimensions") {
            Tensor<float> grad({ 2,3 });

            auto result = ReduceSumToShape(grad, Shape(3));

            CHECK(result.Rank() == 1);
        }

        SUBCASE("ReduceSumToShape same shape") {
            Tensor<float> grad({ 2,3 });

            auto result = ReduceSumToShape(grad, Shape(2, 3));

            CHECK(result.GetShape() == grad.GetShape());
        }

        SUBCASE("ReduceSumToShape throws on incompatible shape") {
            Tensor<float> grad({ 2,3 });

            CHECK_THROWS_AS(ReduceSumToShape(grad, Shape(4, 3)), std::runtime_error);
        }
	}

	TEST_CASE("ExpandToShape Test") {
        SUBCASE("ExpandToShape broadcasts row vector") {
            Tensor<float> input({ 1,3 });

            input[0] = 1;
            input[1] = 2;
            input[2] = 3;

            auto result = ExpandToShape(input, Shape(2, 3));

            CHECK(result[0] == 1);
            CHECK(result[3] == 1);
        }

        SUBCASE("ExpandToShape broadcasts scalar") {
            Tensor<float> scalar({ 1 });
            scalar[0] = 5;

            auto result = ExpandToShape(scalar, Shape(2, 2));

            for (size_t i = 0; i < 4; ++i) {
                CHECK(result[i] == 5);
            }
        }

        SUBCASE("ExpandToShape preserves target shape") {
            Tensor<float> input({ 1, 3 });

            input[0] = 1.0f;
            input[1] = 2.0f;
            input[2] = 3.0f;

            auto result = ExpandToShape(input, Shape(2, 3 ));

            CHECK(result.GetShape() == Shape({ 2, 3 }));

            CHECK(result[0] == doctest::Approx(1.0f));
            CHECK(result[1] == doctest::Approx(2.0f));
            CHECK(result[2] == doctest::Approx(3.0f));

            CHECK(result[3] == doctest::Approx(1.0f));
            CHECK(result[4] == doctest::Approx(2.0f));
            CHECK(result[5] == doctest::Approx(3.0f));
        }
	}
}