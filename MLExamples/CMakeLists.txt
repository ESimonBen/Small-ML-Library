# MLExamples CMakeLists
set(MLEXAMPLES_SOURCES
    main.cpp
)

add_executable(MLExample
    ${MLEXAMPLES_SOURCES}
)

target_link_libraries(MLExample PRIVATE MLCore)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES
    ${MLEXAMPLES_SOURCES}
)