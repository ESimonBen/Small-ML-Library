/// gradFnTests.cpp
#include "dummyGradFn.h"
#include <doctest/doctest.h>
#include <mlCore/operations/operations.h>

using namespace MLCore::Utils;
using namespace MLCore::Memory;
using namespace MLCore::TensorCore;

TEST_SUITE("Base Gradient Function Tests") {
	TEST_CASE("Base Gradient Function Tests") {
		SUBCASE("GradFn stores single input") {
			ArenaAllocator allocator;

			Tensor<float> tensor({ 2, 3 }, allocator);
			auto impl = tensor.GetImpl();

			DummyGradFn<float> gradFn(impl);

			CHECK(gradFn.GetInputs().size() == 1);
			CHECK(gradFn.GetInput(0)->shape == impl->shape);
			CHECK(gradFn.GetInput(0)->allocator == impl->allocator);
			CHECK(gradFn.GetInput(0)->requiresGrad == impl->requiresGrad);
		}

		SUBCASE("GradFn stores multiple input") {
			ArenaAllocator allocator;

			Tensor<float> tensor1({ 2, 3 }, allocator);
			Tensor<float> tensor2({ 2, 3 }, allocator);

			auto impl1 = tensor1.GetImpl();
			auto impl2 = tensor2.GetImpl();

			DummyGradFn<float> gradFn({ impl1, impl2 });

			CHECK(gradFn.GetInputs().size() == 2);

			CHECK(gradFn.GetInput(0)->shape == impl1->shape);
			CHECK(gradFn.GetInput(0)->allocator == impl1->allocator);
			CHECK(gradFn.GetInput(0)->requiresGrad == impl1->requiresGrad);

			CHECK(gradFn.GetInput(1)->shape == impl2->shape);
			CHECK(gradFn.GetInput(1)->allocator == impl2->allocator);
			CHECK(gradFn.GetInput(1)->requiresGrad == impl2->requiresGrad);
		}
	}
}