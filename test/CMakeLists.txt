include(FetchContent)
message(STATUS "Fetching googletest library...")
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 58d77fa8070e8cec2dc1ed015d66b454c8d78850 # release-1.12.1
)
FetchContent_MakeAvailable(googletest)
enable_testing()

file(GLOB TEST_SOURCES *.test.cpp)
add_executable(ALFTest ${TEST_SOURCES})
target_compile_definitions(
        ALFTest
        PUBLIC
        TEST_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(
    ALFTest
    PUBLIC ${PROJECT_NAME}
    PUBLIC GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(ALFTest)

