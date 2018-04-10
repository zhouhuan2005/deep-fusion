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

if(NOT ${WITH_GTEST})
  return()
endif()

# add all *.cc to unit test
file(GLOB TEST_CASES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
list(REMOVE_ITEM TEST_CASES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
set(MAIN_SRC_GTEST main.cc)
set(util_test_cc
  ${PROJECT_SOURCE_DIR}/util/util_test.cc
  ${PROJECT_SOURCE_DIR}/util/util_mkldnn.cc
  ${PROJECT_SOURCE_DIR}/util/util_deepfusion.cc
  ${PROJECT_SOURCE_DIR}/util/util.cc)

foreach(TEST_FILE ${TEST_CASES_SRC})
  get_filename_component(TEST_FILE_NAME ${TEST_FILE} NAME_WE)
  set(EXE_NAME ${TEST_FILE_NAME})
  add_executable(${EXE_NAME} ${MAIN_SRC_GTEST} ${TEST_FILE} ${util_test_cc})
  target_link_libraries(${EXE_NAME} ${LIB_NAME} ${EXTERNAL_LIBS})
  add_dependencies(${EXE_NAME} ${external_project_dependencies})
  add_test(${EXE_NAME} ${EXE_NAME})
endforeach()
