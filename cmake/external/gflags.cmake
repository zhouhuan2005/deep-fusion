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

if(NOT ${WITH_BENCHMARK})
  return()
endif()

include(ExternalProject)

set(GFLAGS_SOURCES_DIR ${THIRD_PARTY_PATH}/gflags)
set(GFLAGS_INSTALL_DIR ${THIRD_PARTY_INSTALL_PATH}/gflags)
set(GFLAGS_INCLUDE_DIR "${GFLAGS_INSTALL_DIR}/include" CACHE PATH "gflags include directory." FORCE)
set(GFLAGS_LIB "${GFLAGS_INSTALL_DIR}/lib/libgflags.a" CACHE FILEPATH "GFLAGS_LIB" FORCE)

include_directories(${GFLAGS_INCLUDE_DIR})

ExternalProject_Add(
    extern_gflags
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY  "https://github.com/gflags/gflags.git"
    GIT_TAG         v2.2.1
    PREFIX          ${GFLAGS_SOURCES_DIR}
    UPDATE_COMMAND  ""
    CMAKE_ARGS      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_INSTALL_PREFIX=${GFLAGS_INSTALL_DIR}
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DBUILD_TESTING=OFF
                    -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${GFLAGS_INSTALL_DIR}
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
)

add_library(gflags STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET gflags PROPERTY IMPORTED_LOCATION ${GFLAGS_LIB})
add_dependencies(gflags extern_gflags)
list(APPEND external_project_dependencies gflags)
