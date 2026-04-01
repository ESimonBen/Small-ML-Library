# MLCore CMakeLists
set(MLCORE_SOURCES
    src/tensor/tensor.cpp
    src/operations/elementwise.cpp
    src/memory/allocator.cpp
    src/utils/shape.cpp
)

set(MLCORE_HEADERS
    include/mlcore/tensor/tensor.h
    include/mlcore/operations/elementwise.h
    include/mlcore/memory/allocator.h
    include/mlcore/utils/shape.h
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