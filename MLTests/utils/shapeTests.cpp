/// shapeTests.cpp
#include <doctest/doctest.h>
#include <mlCore/utils/shape.h>

using namespace MLCore::Utils;

TEST_SUITE("Shape Tests") {
    TEST_CASE("Shape construction from vector") {
        Shape shape({ 2, 3, 4 });

        CHECK(shape.Rank() == 3);
        CHECK(shape.NumElements() == 24);

        CHECK(shape.Dims()[0] == 2);
        CHECK(shape.Dims()[1] == 3);
        CHECK(shape.Dims()[2] == 4);
    }

    TEST_CASE("Shape construction from parameter pack") {
        Shape shape(5, 6);

        CHECK(shape.Rank() == 2);
        CHECK(shape.NumElements() == 30);

        CHECK(shape[0] == 5);
        CHECK(shape[1] == 6);
    }

    TEST_CASE("Empty shape") {
        Shape shape;

        CHECK(shape.Rank() == 0);
        CHECK(shape.NumElements() == 0);

        CHECK(shape.Dims().empty());
        CHECK(shape.Strides().empty());
    }

    TEST_CASE("Shape computes row-major strides") {
        Shape shape({ 2,3,4 });

        auto strides = shape.Strides();

        REQUIRE(strides.size() == 3);

        CHECK(strides[0] == 12);
        CHECK(strides[1] == 4);
        CHECK(strides[2] == 1);
    }

    TEST_CASE("FlattenIndex converts multidimensional indices") {
        Shape shape({ 2,3,4 });

        CHECK(shape.FlattenIndex({ 0,0,0 }) == 0);
        CHECK(shape.FlattenIndex({ 0,0,1 }) == 1);
        CHECK(shape.FlattenIndex({ 0,1,0 }) == 4);
        CHECK(shape.FlattenIndex({ 1,2,3 }) == 23);
    }

    TEST_CASE("UnflattenIndex converts flat indices") {
        Shape shape({ 2,3,4 });

        CHECK(shape.UnflattenIndex(0)
            == std::vector<size_t>{0, 0, 0});

        CHECK(shape.UnflattenIndex(4)
            == std::vector<size_t>{0, 1, 0});

        CHECK(shape.UnflattenIndex(23)
            == std::vector<size_t>{1, 2, 3});
    }

    TEST_CASE("Flatten and unflatten are inverses") {
        Shape shape({ 2,3,4 });

        for (size_t i = 0; i < shape.NumElements(); ++i) {
            auto indices = shape.UnflattenIndex(i);

            CHECK(shape.FlattenIndex(indices) == i);
        }
    }

    TEST_CASE("FlattenIndex throws on dimension mismatch") {
        Shape shape({ 2,3 });

        CHECK_THROWS_AS(
            shape.FlattenIndex({ 1 }),
            std::runtime_error
        );

        CHECK_THROWS_AS(
            shape.FlattenIndex({ 1,2,3 }),
            std::runtime_error
        );
    }

    TEST_CASE("FlattenIndex throws on invalid indices") {
        Shape shape({ 2,3 });

        CHECK_THROWS_AS(
            shape.FlattenIndex({ 2,0 }),
            std::out_of_range
        );

        CHECK_THROWS_AS(
            shape.FlattenIndex({ 0,3 }),
            std::out_of_range
        );
    }

    TEST_CASE("Shape equality") {
        Shape a({ 2,3,4 });
        Shape b({ 2,3,4 });
        Shape c({ 2,4,3 });

        CHECK(a == b);
        CHECK_FALSE(a != b);

        CHECK(a != c);
        CHECK_FALSE(a == c);
    }

    TEST_CASE("Shape copy constructor") {
        Shape original({ 2,3,4 });

        Shape copy(original);

        CHECK(copy == original);
        CHECK(copy.Strides() == original.Strides());
        CHECK(copy.NumElements() == original.NumElements());
    }

    TEST_CASE("Shape copy assignment") {
        Shape original({ 2,3,4 });
        Shape assigned({ 1 });

        assigned = original;

        CHECK(assigned == original);
        CHECK(assigned.Strides() == original.Strides());
    }
}