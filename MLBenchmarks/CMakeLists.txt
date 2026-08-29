# MLBenchmarks CMakeLists
include(FetchContent)

# Prevent Google Benchmark from pulling in googletest
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1 # Choosing a specific tag, will update if needed
)

FetchContent_MakeAvailable(googlebenchmark)

file(GLOB_RECURSE MLBENCH_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_executable(MLBenchmarks ${MLBENCH_SOURCES})

target_link_libraries(MLBenchmarks PRIVATE
    MLCore
    benchmark::benchmark
)

target_compile_features(MLBenchmarks PRIVATE cxx_std_23)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
    FILES ${MLBENCH_SOURCES}
)