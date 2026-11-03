cmake_minimum_required(VERSION 3.15...3.30)
project(specula_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SPECULA_CPP_NATIVE "Optimize for the CPU of the build machine (not portable)" OFF)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# Outside scikit-build-core, ask the nanobind Python package where its CMake files are
if(NOT nanobind_ROOT)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
endif()
find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)

nanobind_add_module(specula_cpp NB_STATIC src/module.cpp)
target_link_libraries(specula_cpp PRIVATE Threads::Threads)

if(SPECULA_CPP_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" SPECULA_HAS_MARCH_NATIVE)
  check_cxx_compiler_flag("-mcpu=native" SPECULA_HAS_MCPU_NATIVE)
  if(SPECULA_HAS_MARCH_NATIVE)
    target_compile_options(specula_cpp PRIVATE -march=native)
  elseif(SPECULA_HAS_MCPU_NATIVE)
    target_compile_options(specula_cpp PRIVATE -mcpu=native)
  endif()
endif()

install(TARGETS specula_cpp LIBRARY DESTINATION .)
