#===============================================================================
# This file is part of the JITInfer (https://github.com/tensor-tangi/jitinfer).
# Copyright (c) 2018 Tensor Tang.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#===============================================================================

cmake_minimum_required(VERSION 3.5)

if(WIN32 OR APPLE)
  message(FATAL "Windows or Mac is not supported yet.")
endif()

# CMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel"
      FORCE)
  message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
endif()

set(PROJECT_NAME "JITInfer")
set(PROJECT_VERSION "0.1")
set(LIB_NAME jitinfer)
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROJECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
  "A path setting third party libraries download & build directories.")
set(THIRD_PARTY_INSTALL_PATH "${THIRD_PARTY_PATH}/install" CACHE STRING
  "A path setting third party libraries install directories.")
set(THIRD_PARTY_BUILD_TYPE Release)

project(${PROJECT_NAME} C CXX)
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
        "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

if (${CMAKE_BUILD_TYPE} MATCHES "Release")
  set(DEBUG_MODE OFF)
else()
  set(DEBUG_MODE ON)
endif()
################################ Configurations #######################################
option(WITH_VNNI          "Compile with AVX512 VNNI support"        ${VNNI_FOUND})  # TODO: enable it when conv
option(WITH_VERBOSE       "Compile with VERBOSE for profiling"      ${DEBUG_MODE})
option(WITH_BENCHMARK     "Compile with benchmark"                             ON)
option(WITH_GTEST         "Compile with gtest"                                 ON)
# enable below option will always allocate about 100MB for clear cache in benchmark
option(WITH_COLD_CACHE    "Benchmark with cold cache method"    ${WITH_BENCHMARK})

# TODO: enable it if necessary
# option(WITH_GLOG          "Compile with GLOG and GFLAG"                       OFF)
########################################################################################
include(system)              # gcc flags
include(simd)                # check if AVX512 or VNNI is supported
include(external/mklml)      # download mklml package and use iomp for jitinfer, use others for mkldnn
include(external/mkldnn)     # download mkldnn for reference comparion
include(external/xbyak)      # download xbyak
include(external/gtest)      # download, build, install gtest

if(NOT ${AVX512_FOUND})
  message(FATAL "AVX512 is not found! Only support on AVX512 yet!")
endif()

if(${CMAKE_BUILD_TYPE} MATCHES "MinSizeRel")
  # disable most features, only generate libjitinfer.so
  set(WITH_VERBOSE OFF)
  set(WITH_BENCHMARK OFF)
  set(WITH_GTEST OFF)
  set(WITH_GLOG OFF)
  set(WITH_COLD_CACHE OFF)
endif()

if(WITH_VERBOSE)
  add_definitions(-DWITH_VERBOSE)
endif()

if(WITH_COLD_CACHE)
  add_definitions(-DWITH_COLD_CACHE)
endif()

if(WITH_GLOG)
  add_definitions(-DWITH_GLOG)
endif()

set(EXTERNAL_LIBS
    ${MKLML_IOMP_LIB}
    gtest
    gtest_main
    ${MKLDNN_LIB}
    ${MKLML_LIB}
)
include_directories("${PROJECT_SOURCE_DIR}")
include_directories(include)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(benchmark)

set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -O0" CACHE STRING "Disable optimize when debug")
set(CMAKE_C_FLAGS_DEBUG "-ggdb3 -O0"  CACHE STRING "Disable optimize when debug")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")

install(FILES LICENSE DESTINATION share/doc/${LIB_NAME})
