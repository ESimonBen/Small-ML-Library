# MLCore CMakeLists
set(MLCORE_SOURCES
    src/mlCore/tensor/tensor.cpp
    src/mlCore/operations/elementwise.cpp
    src/mlCore/memory/allocator.cpp
    src/mlCore/utils/shape.cpp
)

set(MLCORE_HEADERS
    include/mlCore/config.h
    include/mlCore/tensor/tensor.h
    include/mlCore/operations/elementwise.h
    include/mlCore/memory/allocator.h
    include/mlCore/memory/allocator.inl
    include/mlCore/memory/storage.h
    include/mlCore/utils/shape.h
    include/mlCore/utils/shape.inl
)

add_library(MLCore STATIC
    ${MLCORE_SOURCES}
    ${MLCORE_HEADERS}
)

target_compile_features(MLCore PUBLIC cxx_std_23)

target_include_directories(MLCore
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Mirror directory structure in Visual Studio
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES
    ${MLCORE_SOURCES}
    ${MLCORE_HEADERS}
)