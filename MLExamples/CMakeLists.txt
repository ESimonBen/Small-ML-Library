# MLExamples CMakeLists
set(MLEXAMPLES_SOURCES
    main.cpp
)

add_executable(MLExample
    ${MLEXAMPLES_SOURCES}
)

target_link_libraries(MLExample PRIVATE MLCore)

target_compile_features(MLExample PRIVATE cxx_std_23)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES
    ${MLEXAMPLES_SOURCES}
)