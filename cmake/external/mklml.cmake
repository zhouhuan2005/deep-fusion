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

# download mklml package is only for iomp so far
include(ExternalProject)
set(MKLML_PROJECT       "extern_mklml")
set(MKLML_VER           "mklml_lnx_2018.0.2.20180127")
set(MKLML_URL           "https://github.com/01org/mkl-dnn/releases/download/v0.13/${MKLML_VER}.tgz")
set(MKLML_SOURCE_DIR    "${THIRD_PARTY_PATH}/mklml")
set(MKLML_DOWNLOAD_DIR  "${MKLML_SOURCE_DIR}/src/${MKLML_PROJECT}")
set(MKLML_DST_DIR       "mklml")
set(MKLML_INSTALL_ROOT   ${THIRD_PARTY_INSTALL_PATH}/mklml)
set(MKLML_ROOT          ${MKLML_INSTALL_ROOT}/${MKLML_VER})
set(MKLML_INC_DIR       ${MKLML_ROOT}/include)
set(MKLML_LIB_DIR       ${MKLML_ROOT}/lib)
set(MKLML_LIB           ${MKLML_LIB_DIR}/libmklml_intel.so)
set(MKLML_IOMP_LIB      ${MKLML_LIB_DIR}/libiomp5.so)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${MKLML_ROOT}/lib")

include_directories(${MKLML_INC_DIR})

file(WRITE ${MKLML_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(MKLML)\n"
  "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY ${MKLML_VER}\n"
  "        DESTINATION ${MKLML_DST_DIR})\n")

ExternalProject_Add(
    ${MKLML_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX                ${MKLML_SOURCE_DIR}
    DOWNLOAD_DIR          ${MKLML_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate ${MKLML_URL} -c -O ${MKLML_VER}.tgz
                          && tar zxf ${MKLML_VER}.tgz
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
    CMAKE_CACHE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${THIRD_PARTY_INSTALL_PATH}
)

add_library(mklml SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET mklml PROPERTY IMPORTED_LOCATION ${MKLML_IOMP_LIB})
add_dependencies(mklml ${MKLML_PROJECT})
list(APPEND external_project_dependencies mklml)

# set openmp
# find_package(OpenMP), if(OpenMP_C_FOUND), if(OpenMP_CXX_FOUND)
set(OPENMP_FLAGS "-fopenmp")
set(CMAKE_C_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OPENMP_FLAGS})
set(CMAKE_CXX_CREATE_SHARED_LIBRARY_FORBIDDEN_FLAGS ${OPENMP_FLAGS})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPENMP_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_FLAGS}")

# iomp5 must be installed
install(PROGRAMS ${MKLML_IOMP_LIB} DESTINATION lib)
