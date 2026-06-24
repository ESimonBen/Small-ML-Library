/// reductionTests.cpp
#include <doctest/doctest.h>
#include <mlCore/operations/reduction/reduction.h>

using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::TensorCore;
using namespace MLCore::Operations;

TEST_SUITE("Reduction Operations Tests") {
	TEST_CASE("SumAll") {
		SUBCASE("SumAll Calculation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			auto B = SumAll(A, allocator);

			CHECK(B.GetShape() == Shape(1));
			CHECK(B[0] == 21);
		}

		SUBCASE("SumAll on an empty tensor returns 0") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);

			auto B = SumAll(A, allocator);

			CHECK(B.GetShape() == Shape(1));
			CHECK(B[0] == 0);
		}

		SUBCASE("SumAll propagates requires-grad") {
			ArenaAllocator allocator;

			Tensor<float> A({2, 2}, allocator);
			A.Fill(3.0f);
			A.SetRequiresGrad(true);

			auto B = SumAll(A, allocator);

			CHECK(B.RequiresGrad());
		}
	}

	TEST_CASE("MeanAll") {
		SUBCASE("MeanAll Calculation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			auto B = MeanAll(A, allocator);

			CHECK(B.GetShape() == Shape(1));
			CHECK(B[0] == 3.5);
		}

		SUBCASE("MeanAll throws if the tensor is empty") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);

			CHECK_THROWS_AS(MeanAll(A, allocator), std::runtime_error);
		}

		SUBCASE("MeanAll propagates requires-grad") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(3.0f);
			A.SetRequiresGrad(true);

			auto B = MeanAll(A, allocator);

			CHECK(B.RequiresGrad());
		}
	}

	TEST_CASE("MaxAll") {
		SUBCASE("MaxAll Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			auto B = MaxAll(A, allocator);

			CHECK(B.GetShape() == Shape(1));
			CHECK(B[0] == 6);
		}

		SUBCASE("MaxAll throws if the tensor is empty") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);

			CHECK_THROWS_AS(MaxAll(A, allocator), std::runtime_error);
		}

		SUBCASE("MaxAll propagates requires-grad") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(4.0f);
			A.SetRequiresGrad(true);

			auto B = MaxAll(A, allocator);

			CHECK(B.RequiresGrad());
		}
	}

	TEST_CASE("MinAll") {
		SUBCASE("MinAll Operation") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			auto B = MinAll(A, allocator);

			CHECK(B.GetShape() == Shape(1));
			CHECK(B[0] == 1);
		}

		SUBCASE("MinAll throws if the tensor is empty") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);

			CHECK_THROWS_AS(MinAll(A, allocator), std::runtime_error);
		}

		SUBCASE("MinAll propagates requires-grad") {
			ArenaAllocator allocator;
			
			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(5.0f);
			A.SetRequiresGrad(true);

			auto B = MinAll(A, allocator);

			CHECK(B.RequiresGrad());
		}
	}

	TEST_CASE("AxisSum") {
		SUBCASE("AxisSum Calculation (KeepDims)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			auto B = AxisSum(A, 0, allocator, true);

			CHECK(B.GetShape() == Shape(1, 3));
			CHECK(B[0] == 5);
			CHECK(B[1] == 7);
			CHECK(B[2] == 9);
		}

		SUBCASE("AxisSum Calculation (No KeepDims)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			auto B = AxisSum(A, 0, allocator, false);

			CHECK(B.GetShape() == Shape(3));
			CHECK(B[0] == 5);
			CHECK(B[1] == 7);
			CHECK(B[2] == 9);
		}

		SUBCASE("AxisSum with invalid axis throws") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			CHECK_THROWS_AS(AxisSum(A, 2, allocator), std::out_of_range);
		}

		SUBCASE("AxisSum with empty dims throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);

			CHECK_THROWS_AS(AxisSum(A, 0, allocator), std::out_of_range);
		}

		SUBCASE("AxisSum propagates requires-grad") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(3.0f);
			A.SetRequiresGrad(true);

			auto B = AxisSum(A, 1, allocator);

			CHECK(B.RequiresGrad());
		}
	}

	TEST_CASE("AxisMean") {
		SUBCASE("AxisMean Calculation (KeepDims)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			auto B = AxisMean(A, 0, allocator, true);

			CHECK(B.GetShape() == Shape(1, 3));
			CHECK(B[0] == doctest::Approx(2.5f));
			CHECK(B[1] == doctest::Approx(3.5f));
			CHECK(B[2] == doctest::Approx(4.5f));
		}

		SUBCASE("AxisMean Calculation (No KeepDims)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			auto B = AxisMean(A, 0, allocator, false);

			CHECK(B.GetShape() == Shape(3));
			CHECK(B[0] == doctest::Approx(2.5f));
			CHECK(B[1] == doctest::Approx(3.5f));
			CHECK(B[2] == doctest::Approx(4.5f));
		}

		SUBCASE("AxisMean with invalid axis throws") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			CHECK_THROWS_AS(AxisMean(A, 2, allocator), std::out_of_range);
		}

		SUBCASE("AxisMean with empty dims throws") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);

			CHECK_THROWS_AS(AxisMean(A, 0, allocator), std::out_of_range);
		}

		SUBCASE("AxisMean propagates requires-grad") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(5.0f);
			A.SetRequiresGrad(true);

			auto B = AxisMean(A, 0, allocator);

			CHECK(B.RequiresGrad());
		}
	}

	TEST_CASE("AxisMax") {
		SUBCASE("AxisMax Operation (KeepDims)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			auto B = AxisMax(A, 0, allocator, true);

			CHECK(B.GetShape() == Shape(1, 3));
			CHECK(B[0] == 4);
			CHECK(B[1] == 5);
			CHECK(B[2] == 6);
		}

		SUBCASE("AxisMax Operation (No KeepDims)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			auto B = AxisMax(A, 0, allocator, false);

			CHECK(B.GetShape() == Shape(3));
			CHECK(B[0] == 4);
			CHECK(B[1] == 5);
			CHECK(B[2] == 6);
		}

		SUBCASE("AxisMax with invalid axis throws") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			CHECK_THROWS_AS(AxisMax(A, 2, allocator), std::out_of_range);
		}

		SUBCASE("AxisMax with empty dims") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);

			CHECK_THROWS_AS(AxisMax(A, 0, allocator), std::out_of_range);
		}

		SUBCASE("AxisMax propagates requires-grad") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(2.0f);
			A.SetRequiresGrad(true);

			auto B = AxisMax(A, 0, allocator);

			CHECK(B.RequiresGrad());
		}
	}

	TEST_CASE("AxisMin") {
		SUBCASE("AxisMin Operation (KeepDims)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			auto B = AxisMin(A, 0, allocator, true);

			CHECK(B.GetShape() == Shape(1, 3));
			CHECK(B[0] == 1);
			CHECK(B[1] == 2);
			CHECK(B[2] == 3);
		}

		SUBCASE("AxisMin Operation (No KeepDims)") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			auto B = AxisMin(A, 0, allocator, false);

			CHECK(B.GetShape() == Shape(3));
			CHECK(B[0] == 1);
			CHECK(B[1] == 2);
			CHECK(B[2] == 3);
		}

		SUBCASE("AxisMin with invalid axis throws") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			size_t size = A.NumElements();

			for (size_t i = 0; i < size; ++i) {
				A[i] = static_cast<float>(i + 1);
			}

			CHECK_THROWS_AS(AxisMin(A, 2, allocator), std::out_of_range);
		}

		SUBCASE("AxisMin with empty dims") {
			ArenaAllocator allocator;

			Tensor<float> A(Shape(), allocator);

			CHECK_THROWS_AS(AxisMin(A, 0, allocator), std::out_of_range);
		}

		SUBCASE("AxisMin propagates requires-grad") {
			ArenaAllocator allocator;

			Tensor<float> A({ 2, 3 }, allocator);
			A.Fill(2.0f);
			A.SetRequiresGrad(true);

			auto B = AxisMin(A, 0, allocator);

			CHECK(B.RequiresGrad());
		}
	}
}