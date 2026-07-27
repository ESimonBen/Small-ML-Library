/// tensorTests.cpp
#include <doctest/doctest.h>
#include <mlCore/tensor/tensor.h>

using namespace MLCore::Memory;
using namespace MLCore::TensorCore;

TEST_SUITE("Tensor Tests") {
	TEST_CASE("Testing constructor from initializer list") {
		Tensor<float> t({ 2, 3 });

		CHECK(t.Rank() == 2);
		CHECK(t.NumElements() == 6);
		CHECK(t.Dims()[0] == 2);
		CHECK(t.Dims()[1] == 3);
	}

	TEST_CASE("Tensor construction from vector") {
		Tensor<int> t(std::vector<size_t>{4, 5});

		CHECK(t.Rank() == 2);
		CHECK(t.NumElements() == 20);

		CHECK(t.Dims()[0] == 4);
		CHECK(t.Dims()[1] == 5);
	}

	TEST_CASE("Tensor construction from vector") {
		Tensor<int> t(std::vector<size_t>{4, 5});

		CHECK(t.Rank() == 2);
		CHECK(t.NumElements() == 20);

		CHECK(t.Dims()[0] == 4);
		CHECK(t.Dims()[1] == 5);
	}

	TEST_CASE("Linear indexing stores values correctly") {
		Tensor<float> t({ 2,2 });

		t[0] = 1.0f;
		t[1] = 2.0f;
		t[2] = 3.0f;
		t[3] = 4.0f;

		CHECK(t[0] == doctest::Approx(1.0f));
		CHECK(t[1] == doctest::Approx(2.0f));
		CHECK(t[2] == doctest::Approx(3.0f));
		CHECK(t[3] == doctest::Approx(4.0f));
	}

	TEST_CASE("Linear indexing throws when out of bounds") {
		

		Tensor<float> t({ 2,2 });

		CHECK_THROWS_AS(t[4], std::out_of_range);
	}

	TEST_CASE("Vector indexing accesses correct elements") {
		Tensor<int> t({ 2,3 });

		int value = 0;

		for (size_t i = 0; i < t.NumElements(); ++i)
		{
			t[i] = value++;
		}

		CHECK(t({ 0,0 }) == 0);
		CHECK(t({ 0,1 }) == 1);
		CHECK(t({ 0,2 }) == 2);

		CHECK(t({ 1,0 }) == 3);
		CHECK(t({ 1,1 }) == 4);
		CHECK(t({ 1,2 }) == 5);
	}

	TEST_CASE("Variadic indexing accesses correct elements") {
		

		Tensor<int> t({ 2,3 });

		for (size_t i = 0; i < t.NumElements(); ++i)
		{
			t[i] = static_cast<int>(i);
		}

		CHECK(t(0, 0) == 0);
		CHECK(t(0, 2) == 2);

		CHECK(t(1, 0) == 3);
		CHECK(t(1, 2) == 5);
	}

	TEST_CASE("Variadic indexing throws on dimension mismatch") {
		Tensor<int> t({ 2,3 });

		CHECK_THROWS_AS(t(0), std::runtime_error);

		CHECK_THROWS_AS(t(0, 1, 2), std::runtime_error);
	}

	TEST_CASE("Clone performs deep copy") {
		Tensor<float> original({ 2,2 });

		original.Fill(5.0f);

		Tensor<float> copy = original.Clone();

		copy[0] = 100.0f;

		CHECK(original[0] == doctest::Approx(5.0f));

		CHECK(copy[0] == doctest::Approx(100.0f));
	}

	TEST_CASE("Detach shares storage") {
		Tensor<float> original({ 2,2 });

		original.Fill(1.0f);

		Tensor<float> detached = original.Detach();

		detached[0] = 42.0f;

		CHECK(original[0] == doctest::Approx(42.0f));

		CHECK(detached.RequiresGrad() == false);
	}

	TEST_CASE("SliceRows produces a view") {
		Tensor<int> t({ 4,2 });

		for (size_t i = 0; i < t.NumElements(); ++i)
		{
			t[i] = static_cast<int>(i);
		}

		auto slice = t.SliceRows(1, 3);

		CHECK(slice.Dims()[0] == 2);
		CHECK(slice.Dims()[1] == 2);

		CHECK(slice[0] == 2);
		CHECK(slice[1] == 3);
		CHECK(slice[2] == 4);
		CHECK(slice[3] == 5);
	}

	TEST_CASE("SliceRows shares underlying storage") {
		Tensor<int> t({ 4,2 });

		t.Fill(0);

		auto slice = t.SliceRows(1, 3);

		slice[0] = 99;

		CHECK(t[2] == 99);
	}

	TEST_CASE("Concat combines tensors") {
		Tensor<int> a({ 2,2 });
		Tensor<int> b({ 1,2 });

		a[0] = 1;
		a[1] = 2;
		a[2] = 3;
		a[3] = 4;

		b[0] = 5;
		b[1] = 6;

		auto result = Tensor<int>::Concat({ a,b });

		CHECK(result.Dims()[0] == 3);
		CHECK(result.Dims()[1] == 2);

		CHECK(result[0] == 1);
		CHECK(result[1] == 2);
		CHECK(result[2] == 3);
		CHECK(result[3] == 4);
		CHECK(result[4] == 5);
		CHECK(result[5] == 6);
	}

	TEST_CASE("Concat throws on incompatible shapes") {
		Tensor<int> a({ 2,2 });
		Tensor<int> b({ 2,3 });

		CHECK_THROWS_AS(
			Tensor<int>::Concat({ a,b }),
			std::runtime_error
		);
	}

	TEST_CASE("RequiresGrad defaults to false") {
		Tensor<float> t({ 2,2 });

		CHECK_FALSE(t.RequiresGrad());
	}

	TEST_CASE("SetRequiresGrad changes state") {
		Tensor<float> t({ 2,2 });

		t.SetRequiresGrad(true);

		CHECK(t.RequiresGrad());
	}

	TEST_CASE("AccumulateGrad creates gradients") {
		Tensor<float> t({ 2 });

		t.SetRequiresGrad(true);

		Tensor<float> grad({ 2 });

		grad.Fill(1.0f);

		t.AccumulateGrad(grad);

		CHECK(t.HasGrad());

		auto g = t.Grad();

		CHECK(g[0] == doctest::Approx(1.0f));
		CHECK(g[1] == doctest::Approx(1.0f));
	}
}