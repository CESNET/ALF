cmake_minimum_required(VERSION 3.23)
project(
        ALF
        VERSION 1.0
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)
include_directories(include/alf)

find_package(OpenMP REQUIRED)
find_package(mlpack REQUIRED)

add_library(${PROJECT_NAME} STATIC
        include/alf/pool_manager.hpp
        src/pool_manager.cpp include/alf/query.hpp src/query.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC
    OpenMP::OpenMP_CXX
    mlpack::mlpack
)

target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
)

add_subdirectory(test)
