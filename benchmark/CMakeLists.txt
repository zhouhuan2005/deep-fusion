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

file(GLOB BENCHMARK_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

set(TEST_UTILS
  ${PROJECT_SOURCE_DIR}/test/test_utils.cc)

foreach(BENCHMARK_FILE ${BENCHMARK_SRCS})
  get_filename_component(BENCHMARK_FILE_NAME ${BENCHMARK_FILE} NAME_WE)
  set(EXE_NAME ${BENCHMARK_FILE_NAME})
  add_executable(${EXE_NAME} ${BENCHMARK_FILE} ${TEST_UTILS})
  include_directories("${PROJECT_SOURCE_DIR}/test")
  target_link_libraries(${EXE_NAME} ${LIB_NAME}
    ${MKLML_LIB}
    ${MKLDNN_LIB}
    ${GFLAGS_LIB}
    "-L${MKLML_LIB_DIR} -liomp5 -Wl,--as-needed")
  add_dependencies(${EXE_NAME} ${external_project_dependencies})
endforeach()
