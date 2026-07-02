# MLTests CMakeLists
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    *.h
    *.cpp
)

add_executable(MLTests
    ${TEST_SOURCES}
)

target_link_libraries(MLTests
    PRIVATE MLCore
)

target_include_directories(MLTests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/external
)

enable_testing()

add_test(
    NAME MLTests
    COMMAND MLTests
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
    FILES ${TEST_SOURCES}
)