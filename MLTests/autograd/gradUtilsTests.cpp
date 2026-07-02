/// gradUtilsTests.cpp
#include <doctest/doctest.h>
#include <mlCore/autograd/gradientUtils.h>

using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::AutoGrad;
using namespace MLCore::TensorCore;

TEST_SUITE("Gradient Utils Tests") {
	TEST_CASE("ReduceSumToShape") {
        SUBCASE("ReduceSumToShape reduces broadcast dimension") {
            ArenaAllocator allocator;

            Tensor<float> grad({ 2,3 }, allocator);

            grad[0] = 1;
            grad[1] = 2;
            grad[2] = 3;
            grad[3] = 4;
            grad[4] = 5;
            grad[5] = 6;

            auto result = ReduceSumToShape(grad, Shape({ 1,3 }));

            CHECK(result.Dims()[0] == 1);
            CHECK(result[0] == doctest::Approx(5));
            CHECK(result[1] == doctest::Approx(7));
            CHECK(result[2] == doctest::Approx(9));
        }

        SUBCASE("ReduceSumToShape removes leading dimensions") {
            ArenaAllocator allocator;

            Tensor<float> grad({ 2,3 }, allocator);

            auto result = ReduceSumToShape(grad, Shape({ 3 }));

            CHECK(result.Rank() == 1);
        }

        SUBCASE("ReduceSumToShape same shape") {
            ArenaAllocator allocator;

            Tensor<float> grad({ 2,3 }, allocator);

            auto result = ReduceSumToShape(grad, Shape({ 2,3 }));

            CHECK(result.GetShape() == grad.GetShape());
        }

        SUBCASE("ReduceSumToShape throws on incompatible shape") {
            ArenaAllocator allocator;

            Tensor<float> grad({ 2,3 }, allocator);

            CHECK_THROWS_AS(ReduceSumToShape(grad, Shape({ 4,3 })), std::runtime_error);
        }
	}

	TEST_CASE("ExpandToShape") {
        SUBCASE("ExpandToShape broadcasts row vector") {
            ArenaAllocator allocator;

            Tensor<float> input({ 1,3 }, allocator);

            input[0] = 1;
            input[1] = 2;
            input[2] = 3;

            auto result = ExpandToShape(input, Shape({ 2,3 }));

            CHECK(result[0] == 1);
            CHECK(result[3] == 1);
        }

        SUBCASE("ExpandToShape broadcasts scalar") {
            ArenaAllocator allocator;

            Tensor<float> scalar({ 1 }, allocator);
            scalar[0] = 5;

            auto result = ExpandToShape(scalar, Shape({ 2,2 }));

            for (size_t i = 0; i < 4; ++i) {
                CHECK(result[i] == 5);
            }
        }

        SUBCASE("ExpandToShape preserves target shape") {
            ArenaAllocator allocator;

            Tensor<float> input({ 1, 3 }, allocator);

            input[0] = 1.0f;
            input[1] = 2.0f;
            input[2] = 3.0f;

            auto result = ExpandToShape(input, Shape({ 2, 3 }));

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