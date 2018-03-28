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

if(NOT ${WITH_BENCHMARK})
  return()
endif()

# add all *.cc to unit test
file(GLOB BENCHMARK_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
set(COMMON_FILE common.cc)

foreach(TEST_FILE ${BENCHMARK_SRCS})
  get_filename_component(TEST_FILE_NAME ${TEST_FILE} NAME_WE)
  set(EXE_NAME ${TEST_FILE_NAME})
  if(${EXE_NAME} STREQUAL "common")
    continue()
  endif()
  add_executable(${EXE_NAME} ${TEST_FILE} ${COMMON_FILE})
  target_link_libraries(${EXE_NAME} ${LIB_NAME} ${MKLML_LIB} ${MKLDNN_LIB} "-L${MKLML_LIB_DIR} -liomp5 -Wl,--as-needed")
  add_dependencies(${EXE_NAME} ${external_project_dependencies})
endforeach()
