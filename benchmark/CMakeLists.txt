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

if(NOT ${WITH_BENCHMARK})
  return()
endif()

# add all *.cc to unit test
file(GLOB BENCHMARK_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

set(util_bench_cc
  ${PROJECT_SOURCE_DIR}/util/util_benchmark.cc
  ${PROJECT_SOURCE_DIR}/util/util_mkldnn.cc
  ${PROJECT_SOURCE_DIR}/util/util_deepfusion.cc
  ${PROJECT_SOURCE_DIR}/util/util.cc)
foreach(TEST_FILE ${BENCHMARK_SRCS})
  get_filename_component(TEST_FILE_NAME ${TEST_FILE} NAME_WE)
  set(EXE_NAME ${TEST_FILE_NAME})
  add_executable(${EXE_NAME} ${TEST_FILE} ${util_bench_cc})
  target_link_libraries(${EXE_NAME} ${LIB_NAME}
    ${MKLML_LIB}
    ${MKLDNN_LIB}
    ${GFLAGS_LIB}
    "-L${MKLML_LIB_DIR} -liomp5 -Wl,--as-needed")
  add_dependencies(${EXE_NAME} ${external_project_dependencies})
endforeach()
