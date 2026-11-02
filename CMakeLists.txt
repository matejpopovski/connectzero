cmake_minimum_required(VERSION 3.20)
project(connectzero LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_engine
    cpp/bindings.cpp
    cpp/mcts.cpp
    cpp/players.cpp
    cpp/pool.cpp
    cpp/solver.cpp)
target_compile_options(_engine PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>)

find_package(Threads REQUIRED)
target_link_libraries(_engine PRIVATE Threads::Threads)

install(TARGETS _engine LIBRARY DESTINATION connectzero)
