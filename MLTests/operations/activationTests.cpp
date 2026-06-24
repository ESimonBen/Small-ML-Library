/// activationTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/activations/activation.h>

using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Activation Function Tests") {
	TEST_CASE("ReLU") {
        SUBCASE("ReLU Calculation") {
            ArenaAllocator allocator;

            Tensor<float> A({ 5 }, allocator);

            A[0] = -2.0f;
            A[1] = -0.5f;
            A[2] = 0.0f;
            A[3] = 2.0f;
            A[4] = 5.0f;

            auto B = ReLU(A, allocator);

            CHECK(B.GetShape() == A.GetShape());

            CHECK(B[0] == 0.0f);
            CHECK(B[1] == 0.0f);
            CHECK(B[2] == 0.0f);
            CHECK(B[3] == 2.0f);
            CHECK(B[4] == 5.0f);
        }

        SUBCASE("ReLU propagates requires-grad") {
            ArenaAllocator allocator;

            Tensor<float> A({ 3 }, allocator);
            A.SetRequiresGrad(true);

            auto B = ReLU(A, allocator);

            CHECK(B.RequiresGrad());
        }
	}

	TEST_CASE("LeakyReLU") {
        SUBCASE("LeakyReLU Calculation") {
            ArenaAllocator allocator;

            Tensor<float> A({ 4 }, allocator);

            A[0] = -2.0f;
            A[1] = -1.0f;
            A[2] = 1.0f;
            A[3] = 3.0f;

            auto B = LeakyReLU(A, 0.1f, allocator);

            CHECK(B[0] == doctest::Approx(-0.2f));
            CHECK(B[1] == doctest::Approx(-0.1f));
            CHECK(B[2] == doctest::Approx(1.0f));
            CHECK(B[3] == doctest::Approx(3.0f));
        }

        SUBCASE("LeakyReLU propagates requires-grad") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2 }, allocator);
            A.SetRequiresGrad(true);

            auto B = LeakyReLU(A, 0.01f, allocator);

            CHECK(B.RequiresGrad());
        }
	}
	
	TEST_CASE("Sigmoid") {
        SUBCASE("Sigmoid Calculation") {
            ArenaAllocator allocator;

            Tensor<float> A({ 3 }, allocator);

            A[0] = -1.0f;
            A[1] = 0.0f;
            A[2] = 1.0f;

            auto B = Sigmoid(A, allocator);

            CHECK(B[0] == doctest::Approx(0.268941f));
            CHECK(B[1] == doctest::Approx(0.5f));
            CHECK(B[2] == doctest::Approx(0.731059f));
        }

        SUBCASE("Sigmoid preserves shape") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2,3 }, allocator);

            auto B = Sigmoid(A, allocator);

            CHECK(B.GetShape() == A.GetShape());
        }

        SUBCASE("Sigmoid propagates requires-grad") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2 }, allocator);
            A.SetRequiresGrad(true);

            auto B = Sigmoid(A, allocator);

            CHECK(B.RequiresGrad());
        }
	}

	TEST_CASE("Tanh") {
        SUBCASE("Tanh Calculation") {
            ArenaAllocator allocator;

            Tensor<float> A({ 3 }, allocator);

            A[0] = -1.0f;
            A[1] = 0.0f;
            A[2] = 1.0f;

            auto B = Tanh(A, allocator);

            CHECK(B[0] == doctest::Approx(-0.761594f));
            CHECK(B[1] == doctest::Approx(0.0f));
            CHECK(B[2] == doctest::Approx(0.761594f));
        }

        SUBCASE("Tanh propagates requires-grad") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2 }, allocator);
            A.SetRequiresGrad(true);

            auto B = Tanh(A, allocator);

            CHECK(B.RequiresGrad());
        }
	}

	TEST_CASE("Softmax") {
        SUBCASE("Softmax Uniform Input") {
            ArenaAllocator allocator;

            Tensor<float> A({ 3 }, allocator);

            A.Fill(0.0f);

            auto B = Softmax(A, allocator);

            CHECK(B[0] == doctest::Approx(1.0f / 3.0f));
            CHECK(B[1] == doctest::Approx(1.0f / 3.0f));
            CHECK(B[2] == doctest::Approx(1.0f / 3.0f));
        }

        SUBCASE("Softmax probabilities sum to one") {
            ArenaAllocator allocator;

            Tensor<float> A({ 3 }, allocator);

            A[0] = 1.0f;
            A[1] = 2.0f;
            A[2] = 3.0f;

            auto B = Softmax(A, allocator);

            float sum = B[0] + B[1] + B[2];

            CHECK(sum == doctest::Approx(1.0f));
        }

        SUBCASE("Softmax handles large values") {
            ArenaAllocator allocator;

            Tensor<float> A({ 3 }, allocator);

            A[0] = 1000.0f;
            A[1] = 1001.0f;
            A[2] = 1002.0f;

            auto B = Softmax(A, allocator);

            float sum = B[0] + B[1] + B[2];

            CHECK(sum == doctest::Approx(1.0f));

            CHECK(std::isfinite(B[0]));
            CHECK(std::isfinite(B[1]));
            CHECK(std::isfinite(B[2]));
        }

        SUBCASE("Softmax propagates requires-grad") {
            ArenaAllocator allocator;

            Tensor<float> A({ 3 }, allocator);

            A.SetRequiresGrad(true);

            auto B = Softmax(A, allocator);

            CHECK(B.RequiresGrad());
        }
	}

	TEST_CASE("AxisSoftmax") {
        SUBCASE("AxisSoftmax rows sum to one") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2,3 }, allocator);

            for (size_t i = 0; i < 6; i++) {
                A[i] = static_cast<float>(i + 1);
            }

            auto B = AxisSoftmax(A, 1, allocator);

            CHECK(B[0] + B[1] + B[2] == doctest::Approx(1.0f));
            CHECK(B[3] + B[4] + B[5] == doctest::Approx(1.0f));
        }

        SUBCASE("AxisSoftmax throws on invalid axis") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2,3 }, allocator);

            CHECK_THROWS_AS(
                AxisSoftmax(A, 5, allocator),
                std::out_of_range
            );
        }

        SUBCASE("AxisSoftmax propagates requires-grad") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2,3 }, allocator);

            A.SetRequiresGrad(true);

            auto B = AxisSoftmax(A, 1, allocator);

            CHECK(B.RequiresGrad());
        }
	}

	TEST_CASE("LogAxisSoftmax") {
        SUBCASE("Exponentiated log-softmax sums to one") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2,3 }, allocator);

            for (size_t i = 0; i < 6; i++) {
                A[i] = static_cast<float>(i + 1);
            }

            auto B = AxisLogSoftmax(A, 1, allocator);

            float row1 =
                std::exp(B[0]) +
                std::exp(B[1]) +
                std::exp(B[2]);

            float row2 =
                std::exp(B[3]) +
                std::exp(B[4]) +
                std::exp(B[5]);

            CHECK(row1 == doctest::Approx(1.0f));
            CHECK(row2 == doctest::Approx(1.0f));
        }

        SUBCASE("AxisLogSoftmax throws on invalid axis") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2,3 }, allocator);

            CHECK_THROWS_AS(
                AxisLogSoftmax(A, 10, allocator),
                std::out_of_range
            );
        }

        SUBCASE("AxisLogSoftmax propagates requires-grad") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2,3 }, allocator);

            A.SetRequiresGrad(true);

            auto B = AxisLogSoftmax(A, 1, allocator);

            CHECK(B.RequiresGrad());
        }
	}
}