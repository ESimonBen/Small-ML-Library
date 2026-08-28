/// lossTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/loss/loss.h>

using namespace MLCore::Utils;

using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Loss Function Tests") {
	TEST_CASE("MeanSquaredError") {
        SUBCASE("MeanSquaredError - Reduction::None") {
            Tensor<float> pred({ 2, 2 });
            Tensor<float> target({ 2, 2 });

            pred[0] = 1.f; pred[1] = 2.f;
            pred[2] = 3.f; pred[3] = 4.f;

            target[0] = 1.f; target[1] = 0.f;
            target[2] = 5.f; target[3] = 4.f;

            auto result = MeanSquaredError(pred, target, 1, Reduction::None);

            CHECK(result.NumElements() == 2);
            CHECK(result[0] == doctest::Approx(2.0f));
            CHECK(result[1] == doctest::Approx(2.0f));
        }

        SUBCASE("MeanSquaredError - Reduction::Mean") {
            Tensor<float> pred({ 2, 2 });
            Tensor<float> target({ 2, 2 });

            pred[0] = 1.f; pred[1] = 2.f;
            pred[2] = 3.f; pred[3] = 4.f;

            target[0] = 1.f; target[1] = 0.f;
            target[2] = 5.f; target[3] = 4.f;

            auto result = MeanSquaredError(pred, target, 1, Reduction::Mean);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(2.0f));
        }

        SUBCASE("MeanSquaredError - Reduction::Sum") {
            Tensor<float> pred({ 2, 2 });
            Tensor<float> target({ 2, 2 });

            pred[0] = 1.f; pred[1] = 2.f;
            pred[2] = 3.f; pred[3] = 4.f;

            target[0] = 1.f; target[1] = 0.f;
            target[2] = 5.f; target[3] = 4.f;

            auto result = MeanSquaredError(pred, target, 1, Reduction::Sum);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(4.0f));
        }

        SUBCASE("MeanSquaredError - ShapeMismatchThrows") {
            Tensor<float> A({ 2,2 });
            Tensor<float> B({ 3,2 });

            CHECK_THROWS_AS(MeanSquaredError(A, B, 1, Reduction::Mean), std::runtime_error);
        }

        SUBCASE("MeanSquaredError - InvalidAxisThrows") {
            Tensor<float> A({ 2,2 });

            CHECK_THROWS_AS(MeanSquaredError(A, A, 5, Reduction::Mean), std::out_of_range);
        }
	}

	TEST_CASE("MeanAbsoluteError") {
        SUBCASE("MeanAbsoluteError - Reduction::None") {
            Tensor<float> pred({ 2, 2 });
            Tensor<float> target({ 2, 2 });

            pred[0] = 1.f; pred[1] = 2.f;
            pred[2] = 3.f; pred[3] = 4.f;

            target[0] = 1.f; target[1] = 0.f;
            target[2] = 5.f; target[3] = 4.f;

            auto result = MeanAbsoluteError(pred, target, 1, Reduction::None);

            CHECK(result.NumElements() == 2);
            CHECK(result[0] == doctest::Approx(1.0f));
            CHECK(result[1] == doctest::Approx(1.0f));
        }
        
        SUBCASE("MeanAbsoluteError - Reduction::Mean") {
            Tensor<float> pred({ 2, 2 });
            Tensor<float> target({ 2, 2 });

            pred[0] = 1.f; pred[1] = 2.f;
            pred[2] = 3.f; pred[3] = 4.f;

            target[0] = 1.f; target[1] = 0.f;
            target[2] = 5.f; target[3] = 4.f;

            auto result = MeanAbsoluteError(pred, target, 1, Reduction::Mean);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(1.0f));
        }
        
        SUBCASE("MeanAbsoluteError - Reduction::Sum") {
            Tensor<float> pred({ 2, 2 });
            Tensor<float> target({ 2, 2 });

            pred[0] = 1.f; pred[1] = 2.f;
            pred[2] = 3.f; pred[3] = 4.f;

            target[0] = 1.f; target[1] = 0.f;
            target[2] = 5.f; target[3] = 4.f;

            auto result = MeanAbsoluteError(pred, target, 1, Reduction::Sum);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(2.0f));
        }
        
        SUBCASE("MeanAbsoluteError - ShapeMismatchThrows") {
            Tensor<float> A({ 2,2 });
            Tensor<float> B({ 3,2 });

            CHECK_THROWS_AS(MeanAbsoluteError(A, B, 1, Reduction::Mean), std::runtime_error);
        }
        
        SUBCASE("MeanAbsoluteError - InvalidAxisThrows") {
            Tensor<float> A({ 2,2 });

            CHECK_THROWS_AS(MeanAbsoluteError(A, A, 5, Reduction::Mean), std::out_of_range);
        }
	}

    TEST_CASE("BinaryCrossEntropy") {
        SUBCASE("BinaryCrossEntropy - Reduction::None") {
            Tensor<float> pred({ 2 });
            Tensor<float> target({ 2 });

            pred[0] = 0.9f;
            pred[1] = 0.1f;

            target[0] = 1.f;
            target[1] = 0.f;

            auto result = BinaryCrossEntropy(pred, target, 0, Reduction::None);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.10536052));
        }

        SUBCASE("BinaryCrossEntropy - Reduction::Mean") {
            Tensor<float> pred({ 2 });
            Tensor<float> target({ 2 });

            pred[0] = 0.9f;
            pred[1] = 0.1f;

            target[0] = 1.f;
            target[1] = 0.f;

            auto result = BinaryCrossEntropy(pred, target, 0, Reduction::Mean);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.10536052));
        }

        SUBCASE("BinaryCrossEntropy - Reduction::Sum") {
            Tensor<float> pred({ 2 });
            Tensor<float> target({ 2 });

            pred[0] = 0.9f;
            pred[1] = 0.1f;

            target[0] = 1.f;
            target[1] = 0.f;

            auto result = BinaryCrossEntropy(pred, target, 0, Reduction::Sum);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.10536052));
        }

        SUBCASE("BinaryCrossEntropy - Clamp Prevents Infinite Result") {
            Tensor<float> pred({ 2 });
            Tensor<float> target({ 2 });

            pred[0] = 0.0f;
            pred[1] = 1.0f;

            target[0] = 1.0f;
            target[1] = 0.0f;

            auto result = BinaryCrossEntropy(pred, target, 0, Reduction::Mean);

            CHECK(std::isfinite(result[0]));
        }
    }

    TEST_CASE("BinaryCrossEntropyWithLogits") {
        SUBCASE("BinaryCrossEntropyWithLogits - Reduction::None") {
            Tensor<float> logits({ 2 });
            Tensor<float> targets({ 2 });

            logits[0] = -2.0f;
            logits[1] = 2.0f;

            targets[0] = 1.f;
            targets[1] = 0.f;

            auto result = BinaryCrossEntropyWithLogits(logits, targets, 0, Reduction::None);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(2.126928));
        }

        SUBCASE("BinaryCrossEntropyWithLogits - Reduction::Mean") {
            Tensor<float> logits({ 2 });
            Tensor<float> targets({ 2 });

            logits[0] = -2.0f;
            logits[1] = 2.0f;

            targets[0] = 1.f;
            targets[1] = 0.f;

            auto result = BinaryCrossEntropyWithLogits(logits, targets, 0, Reduction::Mean);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(2.126928));
        }

        SUBCASE("BinaryCrossEntropyWithLogits - Reduction::Sum") {
            Tensor<float> logits({ 2 });
            Tensor<float> targets({ 2 });

            logits[0] = -2.0f;
            logits[1] = 2.0f;

            targets[0] = 1.f;
            targets[1] = 0.f;

            auto result = BinaryCrossEntropyWithLogits(logits, targets, 0, Reduction::Sum);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(2.126928));
        }

        SUBCASE("BinaryCrossEntropyWithLogits throws on shape mismatch") {
            Tensor<float> logits({ 2 });
            Tensor<float> targets({ 3 });
            logits.Fill(4.0f);
            targets.Fill(3.0f);

            CHECK_THROWS_AS(BinaryCrossEntropyWithLogits(logits, targets, 0, Reduction::None), std::runtime_error);
        }

        SUBCASE("BinaryCrossEntropyWithLogits throws on invalid axis") {
            Tensor<float> logits({ 2 });
            Tensor<float> targets({ 2 });
            logits.Fill(4.0f);
            targets.Fill(3.0f);

            CHECK_THROWS_AS(BinaryCrossEntropyWithLogits(logits, targets, 5, Reduction::None), std::out_of_range);
        }
    }

    TEST_CASE("CrossEntropy") {
        SUBCASE("CrossEntropy - Reduction::None (No Average Over Classes)") {
            Tensor<float> preds({ 3 });
            Tensor<float> targets({ 3 });

            preds[0] = 0.7f;
            preds[1] = 0.2f;
            preds[2] = 0.1f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropy(preds, targets, 0, Reduction::None);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.356675f));
        }

        SUBCASE("CrossEntropy - Reduction::None (Average Over Classes)") {
            Tensor<float> preds({ 3 });
            Tensor<float> targets({ 3 });

            preds[0] = 0.7f;
            preds[1] = 0.2f;
            preds[2] = 0.1f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropy(preds, targets, 0, Reduction::None, true);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.11889165f));
        }

        SUBCASE("CrossEntropy - Reduction::Mean (No Average Over Classes)") {
            Tensor<float> preds({ 3 });
            Tensor<float> targets({ 3 });

            preds[0] = 0.7f;
            preds[1] = 0.2f;
            preds[2] = 0.1f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropy(preds, targets, 0, Reduction::Mean);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.356675f));
        }

        SUBCASE("CrossEntropy - Reduction::Mean (Average Over Classes)") {
            Tensor<float> preds({ 3 });
            Tensor<float> targets({ 3 });

            preds[0] = 0.7f;
            preds[1] = 0.2f;
            preds[2] = 0.1f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropy(preds, targets, 0, Reduction::Mean, true);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.11889165f));
        }

        SUBCASE("CrossEntropy - Reduction::Sum (No Average Over Classes)") {
            Tensor<float> preds({ 3 });
            Tensor<float> targets({ 3 });

            preds[0] = 0.7f;
            preds[1] = 0.2f;
            preds[2] = 0.1f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropy(preds, targets, 0, Reduction::Sum);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.356675f));
        }

        SUBCASE("CrossEntropy - Reduction::Sum (Average Over Classes)") {
            Tensor<float> preds({ 3 });
            Tensor<float> targets({ 3 });

            preds[0] = 0.7f;
            preds[1] = 0.2f;
            preds[2] = 0.1f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropy(preds, targets, 0, Reduction::Sum, true);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.11889165));
        }

        SUBCASE("CrossEntropy throws on shape mismatch") {
            Tensor<float> preds({ 3 });
            Tensor<float> targets({ 4 });

            preds.Fill(1.0f);
            targets.Fill(9.0f);

            CHECK_THROWS_AS(CrossEntropy(preds, targets, 0, Reduction::None), std::runtime_error);
        }

        SUBCASE("CrossEntropy throws on invalid axis") {
            Tensor<float> preds({ 3 });
            Tensor<float> targets({ 3 });

            preds.Fill(1.0f);
            targets.Fill(9.0f);

            CHECK_THROWS_AS(CrossEntropy(preds, targets, 5, Reduction::None), std::out_of_range);
        }
    }

    TEST_CASE("CrossEntropyWithLogits") {
        SUBCASE("CrossEntropyWithLogits - Reduction::None (No Average Over Classes)") {
            Tensor<float> logits({ 3 });
            Tensor<float> targets({ 3 });

            logits[0] = 2.0f;
            logits[1] = 1.0f;
            logits[2] = 0.0f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropyWithLogits(logits, targets, 0, Reduction::None);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(1.407606f));
        }

        SUBCASE("CrossEntropyWithLogits - Reduction::None (Average Over Classes)") {
            Tensor<float> logits({ 3 });
            Tensor<float> targets({ 3 });

            logits[0] = 2.0f;
            logits[1] = 1.0f;
            logits[2] = 0.0f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropyWithLogits(logits, targets, 0, Reduction::None, true);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.469202f));
        }

        SUBCASE("CrossEntropyWithLogits - Reduction::Mean (No Average Over Classes)") {
            Tensor<float> logits({ 3 });
            Tensor<float> targets({ 3 });

            logits[0] = 2.0f;
            logits[1] = 1.0f;
            logits[2] = 0.0f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropyWithLogits(logits, targets, 0, Reduction::Mean);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(1.407606f));
        }

        SUBCASE("CrossEntropyWithLogits - Reduction::Mean (Average Over Classes)") {
            Tensor<float> logits({ 3 });
            Tensor<float> targets({ 3 });

            logits[0] = 2.0f;
            logits[1] = 1.0f;
            logits[2] = 0.0f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropyWithLogits(logits, targets, 0, Reduction::Mean, true);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.469202f));
        }

        SUBCASE("CrossEntropyWithLogits - Reduction::Sum (No Average Over Classes)") {
            Tensor<float> logits({ 3 });
            Tensor<float> targets({ 3 });

            logits[0] = 2.0f;
            logits[1] = 1.0f;
            logits[2] = 0.0f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropyWithLogits(logits, targets, 0, Reduction::Sum);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(1.407606f));
        }

        SUBCASE("CrossEntropyWithLogits - Reduction::Sum (Average Over Classes)") {
            Tensor<float> logits({ 3 });
            Tensor<float> targets({ 3 });

            logits[0] = 2.0f;
            logits[1] = 1.0f;
            logits[2] = 0.0f;

            targets[0] = 1.0f;
            targets[1] = 0.0f;
            targets[2] = 0.0f;

            auto result = CrossEntropyWithLogits(logits, targets, 0, Reduction::Sum, true);

            CHECK(result.NumElements() == 1);
            CHECK(result[0] == doctest::Approx(0.469202f));
        }

        SUBCASE("CrossEntropyWithLogits throws on shape mismatch") {
            Tensor<float> logits({ 3 });
            Tensor<float> targets({ 4 });

            logits.Fill(1.0f);
            targets.Fill(9.0f);

            CHECK_THROWS_AS(CrossEntropyWithLogits(logits, targets, 0, Reduction::None), std::runtime_error);
        }

        SUBCASE("CrossEntropyWithLogits throws on invalid axis") {
            Tensor<float> logits({ 3 });
            Tensor<float> targets({ 3 });

            logits.Fill(1.0f);
            targets.Fill(9.0f);

            CHECK_THROWS_AS(CrossEntropyWithLogits(logits, targets, 5, Reduction::None), std::out_of_range);
        }
    }
}