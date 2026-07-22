/// gradUtilsTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/broadcast/broadcast.h>

using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::Operations;
using namespace MLCore::TensorCore;

TEST_SUITE("Broadcast Gradient Tests") {
    TEST_CASE("Squeeze Gradient") {
        SUBCASE("Squeeze Gradient Operation") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2, 1 }, allocator);
            A.Fill(3.0f);
            A.SetRequiresGrad(true);

            auto B = Squeeze(A, 1, allocator);
            CHECK(B.GetShape() == Shape(2));
            CHECK(B.RequiresGrad());

            auto loss = SumAll(B, allocator);
            loss.Backward();

            auto gradA = A.Grad();
            CHECK(gradA.GetShape() == Shape(2, 1));

            for (auto& val : gradA) {
                CHECK(val == 1.0f);
            }
        }

        SUBCASE("Empty tensor Squeeze throws") {
            ArenaAllocator allocator;

            Tensor<float> A(Shape(), allocator);

            CHECK_THROWS_AS(Squeeze(A, 1, allocator), std::out_of_range);
        }
    }

    TEST_CASE("Unsqueeze Gradient") {
        SUBCASE("Unsqueeze Gradient Operation") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2, 3 }, allocator);
            A.Fill(7.0f);
            A.SetRequiresGrad(true);

            auto B = Unsqueeze(A, 0, allocator);
            CHECK(B.GetShape() == Shape(1, 2, 3));
            CHECK(B.RequiresGrad());

            auto loss = SumAll(B, allocator);
            loss.Backward();

            auto gradA = A.Grad();

            CHECK(gradA.GetShape() == Shape(2, 3));

            for (auto& val : gradA) {
                CHECK(val == 1.0f);
            }
        }

        SUBCASE("Empty tensor Unsqueeze throws") {
            ArenaAllocator allocator;

            Tensor<float> A(Shape(), allocator);

            CHECK_THROWS_AS(Unsqueeze(A, 1, allocator), std::out_of_range);
        }
    }

	TEST_CASE("ReduceSumToShape Gradient") {
        SUBCASE("ReduceSumToShape Gradient Operation") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2, 3 }, allocator);
            A.Fill(2.0f);
            A.SetRequiresGrad(true);

            auto B = ReduceSumToShape(A, Shape(2, 1), allocator);
            CHECK(B.GetShape() == Shape(2, 1));
            CHECK(B.RequiresGrad());

            auto loss = SumAll(B, allocator);
            loss.Backward();

            auto gradA = A.Grad();
            
            CHECK(gradA.GetShape() == Shape(2, 3));

            for (auto& val : gradA) {
                CHECK(val == 1.0f);
            }
        }

        SUBCASE("Empty tensor ReduceSumToShape throws") {
            ArenaAllocator allocator;

            Tensor<float> A(Shape(), allocator);
            A.SetRequiresGrad(true);

            CHECK_THROWS_AS(ReduceSumToShape(A, Shape(1), allocator), std::runtime_error);
        }
	}

	TEST_CASE("ExpandToShape Gradient") {
        SUBCASE("ExpandToShape Gradient Operation") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2, 1 }, allocator);
            A.Fill(2.0f);
            A.SetRequiresGrad(true);

            auto B = ExpandToShape(A, Shape(2, 5), allocator);
            CHECK(B.GetShape() == Shape(2, 5));
            CHECK(B.RequiresGrad());

            auto loss = SumAll(B, allocator);
            loss.Backward();

            auto gradA = A.Grad();

            CHECK(gradA.GetShape() == Shape(2, 1));

            for (auto& val : gradA) {
                CHECK(val == 5.0f);
            }
        }

        SUBCASE("Empty tensor ExpandToShape throws") {
            ArenaAllocator allocator;

            Tensor<float> A(Shape(), allocator);
            A.SetRequiresGrad(true);

            CHECK_THROWS_AS(ExpandToShape(A, Shape(1), allocator), std::runtime_error);
        }
	}
}