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
        CHECK(copy.NumElements() == original.NumElements());
    }

    TEST_CASE("Shape copy assignment") {
        Shape original({ 2,3,4 });
        Shape assigned({ 1 });

        assigned = original;

        CHECK(assigned == original);
    }
}