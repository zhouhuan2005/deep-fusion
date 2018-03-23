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

set(MAIN_SRC_GTEST main.cc)
set(EXE_NAME test_concat)
add_executable(${EXE_NAME} ${MAIN_SRC_GTEST} test_concat.cc)
target_link_libraries(${EXE_NAME} ${LIB_NAME} ${EXTERNAL_LIBS})
add_dependencies(${EXE_NAME} ${external_project_dependencies})
add_test(${EXE_NAME} ${EXE_NAME})
