# This file is part of the JITInfer distribution (https://github.com/tensor-tang).
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
  ${PROJECT_SOURCE_DIR}/util/util_jitinfer.cc
  ${PROJECT_SOURCE_DIR}/util/util.cc)

foreach(TEST_FILE ${TEST_CASES_SRC})
  get_filename_component(TEST_FILE_NAME ${TEST_FILE} NAME_WE)
  set(EXE_NAME ${TEST_FILE_NAME})
  add_executable(${EXE_NAME} ${MAIN_SRC_GTEST} ${TEST_FILE} ${util_test_cc})
  target_link_libraries(${EXE_NAME} ${LIB_NAME} ${EXTERNAL_LIBS})
  add_dependencies(${EXE_NAME} ${external_project_dependencies})
  add_test(${EXE_NAME} ${EXE_NAME})
endforeach()
