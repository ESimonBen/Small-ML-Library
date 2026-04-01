# MLTest CMakeLists
set(MLTEST_SOURCES
    main.cpp
)

add_executable(MLTest
    ${MLTEST_SOURCES}
)

target_link_libraries(MLTest PRIVATE MLCore)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES
    ${MLTEST_SOURCES}
)