# MLCore CMakeLists
set(MLCORE_SOURCES
    src/mlcore/tensor/tensor.cpp
    src/mlcore/operations/elementwise.cpp
    src/mlcore/memory/allocator.cpp
    src/mlcore/utils/shape.cpp
)

set(MLCORE_HEADERS
    include/mlcore/config.h
    include/mlcore/tensor/tensor.h
    include/mlcore/operations/elementwise.h
    include/mlcore/memory/allocator.h
    include/mlcore/memory/allocator.inl
    include/mlcore/memory/storage.h
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