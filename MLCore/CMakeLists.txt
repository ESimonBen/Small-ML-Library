# MLCore CMakeLists
file(GLOB_RECURSE MLCORE_SOURCES CONFIGURE_DEPENDS
    src/*.cpp
)

file(GLOB_RECURSE MLCORE_HEADERS CONFIGURE_DEPENDS
    include/*.h
    include/*.inl
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