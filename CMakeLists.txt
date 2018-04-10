#===============================================================================
# Copyright 2016-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

cmake_minimum_required(VERSION 3.5)

if(WIN32 OR APPLE)
  message(FATAL_ERROR "Windows or Mac is not supported yet.")
endif()

# CMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel"
      FORCE)
  message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
endif()

set(PROJECT_NAME "DeepFusion")
set(PROJECT_VERSION "0.1")
set(LIB_NAME deepfusion)
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

if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS "5.4")
  message(WARNING "The CXX compiler version: ${CMAKE_C_COMPILER_ID}"
        "${CMAKE_C_COMPILER_VERSION} is lower than 5.4, may have some unkown issues")
endif()

if (${CMAKE_BUILD_TYPE} MATCHES "Release")
  set(DEBUG_MODE OFF)
else()
  set(DEBUG_MODE ON)
endif()

################################ Configurations #######################################
option(WITH_VNNI          "Compile with AVX512 VNNI support"          ${VNNI_FOUND})  # TODO: enable it when conv
option(WITH_VERBOSE       "Compile with VERBOSE for profiling"        ${DEBUG_MODE})
option(WITH_DUMP_CODE     "Compile with enabling dump code from JIT"  ${DEBUG_MODE})
option(WITH_BENCHMARK     "Compile with benchmark"                               ON)
option(WITH_GTEST         "Compile with gtest"                                   ON)
# enable below option will always allocate about 100MB for clear cache in benchmark
option(WITH_COLD_CACHE    "Benchmark with cold cache method"      ${WITH_BENCHMARK})
# TODO: enable it if necessary
# option(WITH_GLOG          "Compile with GLOG and GFLAG"                       OFF)
########################################################################################
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include(system)              # gcc flags
include(simd)                # check if AVX512 or VNNI is supported
include(external/mklml)      # download mklml package and use iomp, use others for mkldnn
include(external/mkldnn)     # download mkldnn for reference comparion
include(external/xbyak)      # download xbyak
include(external/gtest)      # download, build, install gtest
include(external/gflags)     # download, build, install gflags, only used for benchmark command line

if(NOT AVX512_FOUND)
  message(FATAL_ERROR "AVX512 is not found! Only support on AVX512 yet!")
endif()

if(${CMAKE_BUILD_TYPE} MATCHES "MinSizeRel")
  # disable most features, only generate libdeepfusion.so
  set(WITH_VERBOSE OFF)
  set(WITH_DUMP_CODE OFF)
  set(WITH_BENCHMARK OFF)
  set(WITH_GTEST OFF)
  set(WITH_GLOG OFF)
  set(WITH_COLD_CACHE OFF)
endif()

if(WITH_VERBOSE)
  add_definitions(-DWITH_VERBOSE)
endif()

if(WITH_DUMP_CODE)
  add_definitions(-DWITH_DUMP_CODE)
endif()

if(WITH_COLD_CACHE)
  add_definitions(-DWITH_COLD_CACHE)
endif()

#if(WITH_GLOG)
#  add_definitions(-DWITH_GLOG)
#endif()

set(EXTERNAL_LIBS
    ${MKLML_IOMP_LIB}
    gtest
    gtest_main
    ${MKLDNN_LIB}
    ${MKLML_LIB}
)
include_directories("${PROJECT_SOURCE_DIR}")
include_directories(include)
include_directories(util)

# TODO: util::_getenv has warnning
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-stringop-overflow")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow")

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(benchmark)

set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3 -O0" CACHE STRING "Disable optimize when debug")
set(CMAKE_C_FLAGS_DEBUG "-ggdb3 -O0"  CACHE STRING "Disable optimize when debug")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")

install(FILES LICENSE DESTINATION share/doc/${LIB_NAME})
