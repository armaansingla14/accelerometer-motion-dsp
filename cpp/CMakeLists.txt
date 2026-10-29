cmake_minimum_required(VERSION 3.16)
project(motiondsp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

add_library(motiondsp src/filters.cpp src/fft.cpp src/features.cpp src/window_io.cpp)
target_include_directories(motiondsp PUBLIC include)

add_executable(extract_features tools/extract_features.cpp)
target_link_libraries(extract_features PRIVATE motiondsp)

add_executable(bench_latency bench/bench_latency.cpp)
target_link_libraries(bench_latency PRIVATE motiondsp)

enable_testing()
add_executable(motiondsp_tests tests/test_main.cpp tests/test_filters.cpp tests/test_fft.cpp tests/test_features.cpp)
target_link_libraries(motiondsp_tests PRIVATE motiondsp)
add_test(NAME motiondsp_tests COMMAND motiondsp_tests)
