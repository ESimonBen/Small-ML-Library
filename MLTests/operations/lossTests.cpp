/// lossTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/loss/loss.h>

using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Loss Function Tests") {
	TEST_CASE("MeanSquaredError") {
        SUBCASE("MeanSquaredError - Reduction::None") {
            ArenaAllocator allocator;

            Tensor<float> pred({ 2, 2 }, allocator);
            Tensor<float> target({ 2, 2 }, allocator);

            pred[0] = 1.f; pred[1] = 2.f;
            pred[2] = 3.f; pred[3] = 4.f;

            target[0] = 1.f; target[1] = 0.f;
            target[2] = 5.f; target[3] = 4.f;

            auto result = MeanSquaredError(pred, target, 1, Reduction::None, allocator);

            CHECK(result.NumElements() == 2);
            CHECK(result[0] == doctest::Approx(2.0f));
            CHECK(result[1] == doctest::Approx(2.0f));
        }

        SUBCASE("MeanSquaredError - Reduction::Mean") {
            ArenaAllocator allocator;

            Tensor<float> pred({ 2, 2 }, allocator);
            Tensor<float> target({ 2, 2 }, allocator);

            pred[0] = 1.f; pred[1] = 2.f;
            pred[2] = 3.f; pred[3] = 4.f;

            target[0] = 1.f; target[1] = 0.f;
            target[2] = 5.f; target[3] = 4.f;

            auto result = MeanSquaredError(pred, target, 1, Reduction::Mean, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(2.0f));
        }

        SUBCASE("MeanSquaredError - Reduction::Sum") {
            ArenaAllocator allocator;

            Tensor<float> pred({ 2, 2 }, allocator);
            Tensor<float> target({ 2, 2 }, allocator);

            pred[0] = 1.f; pred[1] = 2.f;
            pred[2] = 3.f; pred[3] = 4.f;

            target[0] = 1.f; target[1] = 0.f;
            target[2] = 5.f; target[3] = 4.f;

            auto result = MeanSquaredError(pred, target, 1, Reduction::Sum, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(4.0f));
        }

        SUBCASE("MeanSquaredError - ShapeMismatchThrows") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2,2 }, allocator);
            Tensor<float> B({ 3,2 }, allocator);

            CHECK_THROWS_AS(MeanSquaredError(A, B, 1, Reduction::Mean, allocator), std::runtime_error);
        }

        SUBCASE("MeanSquaredError - InvalidAxisThrows") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2,2 }, allocator);

            CHECK_THROWS_AS(MeanSquaredError(A, A, 5, Reduction::Mean, allocator), std::out_of_range);
        }
	}

	TEST_CASE("MeanAbsoluteError") {
        SUBCASE("MeanAbsoluteError - Reduction::None") {
            ArenaAllocator allocator;

            Tensor<float> pred({ 2, 2 }, allocator);
            Tensor<float> target({ 2, 2 }, allocator);

            pred[0] = 1.f; pred[1] = 2.f;
            pred[2] = 3.f; pred[3] = 4.f;

            target[0] = 1.f; target[1] = 0.f;
            target[2] = 5.f; target[3] = 4.f;

            auto result = MeanAbsoluteError(pred, target, 1, Reduction::None, allocator);

            CHECK(result.NumElements() == 2);
            CHECK(result[0] == doctest::Approx(1.0f));
            CHECK(result[1] == doctest::Approx(1.0f));
        }
        
        SUBCASE("MeanAbsoluteError - Reduction::Mean") {
            ArenaAllocator allocator;

            Tensor<float> pred({ 2, 2 }, allocator);
            Tensor<float> target({ 2, 2 }, allocator);

            pred[0] = 1.f; pred[1] = 2.f;
            pred[2] = 3.f; pred[3] = 4.f;

            target[0] = 1.f; target[1] = 0.f;
            target[2] = 5.f; target[3] = 4.f;

            auto result = MeanAbsoluteError(pred, target, 1, Reduction::Mean, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(1.0f));
        }
        
        SUBCASE("MeanAbsoluteError - Reduction::Sum") {
            ArenaAllocator allocator;

            Tensor<float> pred({ 2, 2 }, allocator);
            Tensor<float> target({ 2, 2 }, allocator);

            pred[0] = 1.f; pred[1] = 2.f;
            pred[2] = 3.f; pred[3] = 4.f;

            target[0] = 1.f; target[1] = 0.f;
            target[2] = 5.f; target[3] = 4.f;

            auto result = MeanAbsoluteError(pred, target, 1, Reduction::Sum, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(2.0f));
        }
        
        SUBCASE("MeanAbsoluteError - ShapeMismatchThrows") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2,2 }, allocator);
            Tensor<float> B({ 3,2 }, allocator);

            CHECK_THROWS_AS(MeanAbsoluteError(A, B, 1, Reduction::Mean, allocator), std::runtime_error);
        }
        
        SUBCASE("MeanAbsoluteError - InvalidAxisThrows") {
            ArenaAllocator allocator;

            Tensor<float> A({ 2,2 }, allocator);

            CHECK_THROWS_AS(MeanAbsoluteError(A, A, 5, Reduction::Mean, allocator), std::out_of_range);
        }
	}

    TEST_CASE("BinaryCrossEntropy") {
        SUBCASE("BinaryCrossEntropy - Reduction::None") {
            ArenaAllocator allocator;

            Tensor<float> pred({ 2 }, allocator);
            Tensor<float> target({ 2 }, allocator);

            pred[0] = 0.9f;
            pred[1] = 0.1f;

            target[0] = 1.f;
            target[1] = 0.f;

            auto result = BinaryCrossEntropy(pred, target, 0, Reduction::None, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.10536052));
        }

        SUBCASE("BinaryCrossEntropy - Reduction::Mean") {
            ArenaAllocator allocator;

            Tensor<float> pred({ 2 }, allocator);
            Tensor<float> target({ 2 }, allocator);

            pred[0] = 0.9f;
            pred[1] = 0.1f;

            target[0] = 1.f;
            target[1] = 0.f;

            auto result = BinaryCrossEntropy(pred, target, 0, Reduction::Mean, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.10536052));
        }

        SUBCASE("BinaryCrossEntropy - Reduction::Sum") {
            ArenaAllocator allocator;

            Tensor<float> pred({ 2 }, allocator);
            Tensor<float> target({ 2 }, allocator);

            pred[0] = 0.9f;
            pred[1] = 0.1f;

            target[0] = 1.f;
            target[1] = 0.f;

            auto result = BinaryCrossEntropy(pred, target, 0, Reduction::Sum, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.10536052));
        }

        SUBCASE("BinaryCrossEntropy - ClampPreventsInf") {
            ArenaAllocator allocator;

            Tensor<float> pred({ 2 }, allocator);
            Tensor<float> target({ 2 }, allocator);

            pred[0] = 0.0f;
            pred[1] = 1.0f;

            target[0] = 1.0f;
            target[1] = 0.0f;

            auto result = BinaryCrossEntropy(pred, target, 0, Reduction::Mean, allocator);

            CHECK(std::isfinite(result[0]));
        }
    }

    TEST_CASE("BinaryCrossEntropyWithLogits") {
        SUBCASE("BinaryCrossEntropyWithLogits - Reduction::None") {
            ArenaAllocator allocator;

            Tensor<float> logits({ 2 }, allocator);
            Tensor<float> targets({ 2 }, allocator);

            logits[0] = -2.0f;
            logits[1] = 2.0f;

            targets[0] = 1.f;
            targets[1] = 0.f;

            auto result = BinaryCrossEntropyWithLogits(logits, targets, 0, Reduction::None, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(2.126928));
        }

        SUBCASE("BinaryCrossEntropyWithLogits - Reduction::Mean") {
            ArenaAllocator allocator;

            Tensor<float> logits({ 2 }, allocator);
            Tensor<float> targets({ 2 }, allocator);

            logits[0] = -2.0f;
            logits[1] = 2.0f;

            targets[0] = 1.f;
            targets[1] = 0.f;

            auto result = BinaryCrossEntropyWithLogits(logits, targets, 0, Reduction::Mean, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(2.126928));
        }

        SUBCASE("BinaryCrossEntropyWithLogits - Reduction::Sum") {
            ArenaAllocator allocator;

            Tensor<float> logits({ 2 }, allocator);
            Tensor<float> targets({ 2 }, allocator);

            logits[0] = -2.0f;
            logits[1] = 2.0f;

            targets[0] = 1.f;
            targets[1] = 0.f;

            auto result = BinaryCrossEntropyWithLogits(logits, targets, 0, Reduction::Sum, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(2.126928));
        }

        SUBCASE("BinaryCrossEntropyWithLogits throws on shape mismatch") {
            ArenaAllocator allocator;

            Tensor<float> logits({ 2 }, allocator);
            Tensor<float> targets({ 3 }, allocator);
            logits.Fill(4.0f);
            targets.Fill(3.0f);

            CHECK_THROWS_AS(BinaryCrossEntropyWithLogits(logits, targets, 0, Reduction::None, allocator), std::runtime_error);
        }

        SUBCASE("BinaryCrossEntropyWithLogits throws on invalid axis") {
            ArenaAllocator allocator;

            Tensor<float> logits({ 2 }, allocator);
            Tensor<float> targets({ 2 }, allocator);
            logits.Fill(4.0f);
            targets.Fill(3.0f);

            CHECK_THROWS_AS(BinaryCrossEntropyWithLogits(logits, targets, 5, Reduction::None, allocator), std::out_of_range);
        }
    }

    TEST_CASE("CrossEntropy") {
        SUBCASE("CrossEntropy - Reduction::None") {
            ArenaAllocator allocator;

            Tensor<float> preds({ 3 }, allocator);
            Tensor<float> targets({ 3 }, allocator);

            preds[0] = 0.7f;
            preds[1] = 0.2f;
            preds[2] = 0.1f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropy(preds, targets, 0, Reduction::None, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.11889165));
        }

        SUBCASE("CrossEntropy - Reduction::Mean") {
            ArenaAllocator allocator;

            Tensor<float> preds({ 3 }, allocator);
            Tensor<float> targets({ 3 }, allocator);

            preds[0] = 0.7f;
            preds[1] = 0.2f;
            preds[2] = 0.1f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropy(preds, targets, 0, Reduction::Mean, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.11889165));
        }

        SUBCASE("CrossEntropy - Reduction::Sum") {
            ArenaAllocator allocator;

            Tensor<float> preds({ 3 }, allocator);
            Tensor<float> targets({ 3 }, allocator);

            preds[0] = 0.7f;
            preds[1] = 0.2f;
            preds[2] = 0.1f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropy(preds, targets, 0, Reduction::Sum, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.11889165));
        }

        SUBCASE("CrossEntropy throws on shape mismatch") {
            ArenaAllocator allocator;

            Tensor<float> preds({ 3 }, allocator);
            Tensor<float> targets({ 4 }, allocator);

            preds.Fill(1.0f);
            targets.Fill(9.0f);

            CHECK_THROWS_AS(CrossEntropy(preds, targets, 0, Reduction::None, allocator), std::runtime_error);
        }

        SUBCASE("CrossEntropy throws on invalid axis") {
            ArenaAllocator allocator;

            Tensor<float> preds({ 3 }, allocator);
            Tensor<float> targets({ 3 }, allocator);

            preds.Fill(1.0f);
            targets.Fill(9.0f);

            CHECK_THROWS_AS(CrossEntropy(preds, targets, 5, Reduction::None, allocator), std::out_of_range);
        }
    }

    TEST_CASE("CrossEntropyWithLogits") {
        SUBCASE("CrossEntropyWithLogits - Reduction::None") {
            ArenaAllocator allocator;

            Tensor<float> logits({ 3 }, allocator);
            Tensor<float> targets({ 3 }, allocator);

            logits[0] = 2.0f;
            logits[1] = 1.0f;
            logits[2] = 0.0f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropyWithLogits(logits, targets, 0, Reduction::None, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.13586865));
        }

        SUBCASE("CrossEntropyWithLogits - Reduction::Mean") {
            ArenaAllocator allocator;

            Tensor<float> logits({ 3 }, allocator);
            Tensor<float> targets({ 3 }, allocator);

            logits[0] = 2.0f;
            logits[1] = 1.0f;
            logits[2] = 0.0f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropyWithLogits(logits, targets, 0, Reduction::Mean, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.13586865));
        }

        SUBCASE("CrossEntropyWithLogits - Reduction::Sum") {
            ArenaAllocator allocator;

            Tensor<float> logits({ 3 }, allocator);
            Tensor<float> targets({ 3 }, allocator);

            logits[0] = 2.0f;
            logits[1] = 1.0f;
            logits[2] = 0.0f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropyWithLogits(logits, targets, 0, Reduction::Sum, allocator);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.13586865));
        }

        SUBCASE("CrossEntropyWithLogits throws on shape mismatch") {
            ArenaAllocator allocator;

            Tensor<float> logits({ 3 }, allocator);
            Tensor<float> targets({ 4 }, allocator);

            logits.Fill(1.0f);
            targets.Fill(9.0f);

            CHECK_THROWS_AS(CrossEntropyWithLogits(logits, targets, 0, Reduction::None, allocator), std::runtime_error);
        }

        SUBCASE("CrossEntropyWithLogits throws on invalid axis") {
            ArenaAllocator allocator;

            Tensor<float> logits({ 3 }, allocator);
            Tensor<float> targets({ 3 }, allocator);

            logits.Fill(1.0f);
            targets.Fill(9.0f);

            CHECK_THROWS_AS(CrossEntropyWithLogits(logits, targets, 5, Reduction::None, allocator), std::out_of_range);
        }
    }
}