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

set(TARGET_NAME ${LIB_NAME})

file(GLOB_RECURSE HEADERS
    ${PROJECT_SOURCE_DIR}/include/*.h
    ${PROJECT_SOURCE_DIR}/include/*.hpp
    )
file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    )
set(util_jitinfer_cc
  ${PROJECT_SOURCE_DIR}/util/util_jitinfer.cc
  ${PROJECT_SOURCE_DIR}/util/util.cc)
add_library(${TARGET_NAME} SHARED ${HEADERS} ${SOURCES} ${util_jitinfer_cc})
add_dependencies(${TARGET_NAME} ${external_project_dependencies})

target_link_libraries(${TARGET_NAME} "-L${MKLML_LIB_DIR} -liomp5 -Wl,--as-needed")
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${TARGET_NAME} PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET ${TARGET_NAME} PROPERTY SOVERSION "0")

install(TARGETS ${TARGET_NAME} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include)
