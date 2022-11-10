cmake_minimum_required(VERSION 3.22)
project(
        ALF
        VERSION 1.0
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)
include_directories(include/alf)

add_library(${PROJECT_NAME} STATIC
        include/alf/state_manager.hpp
        src/state_manager.cpp

        include/alf/query.hpp
        src/query.cpp

        include/alf/strategy.hpp
        src/strategy.cpp

        include/alf/model.hpp
        src/model.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC
        -fopenmp
        -larmadillo
        -lsqlite3)

target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>)

add_subdirectory(test)
