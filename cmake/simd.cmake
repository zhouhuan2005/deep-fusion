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

# This file is use to check VNNI support and all other level of AVX on your machine

include(CheckCXXSourceRuns)
include(CheckCXXSourceCompiles)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(MMX_FLAG "-mmmx")
    set(SSE2_FLAG "-msse2")
    set(SSE3_FLAG "-msse3")
    set(AVX_FLAG "-mavx")
    set(AVX2_FLAG "-mavx2")
    set(AVX512_FLAG "-mavx512f")
    set(VNNI_FLAG "-mavx512vnni")
elseif(MSVC)
    set(MMX_FLAG "/arch:MMX")
    set(SSE2_FLAG "/arch:SSE2")
    set(SSE3_FLAG "/arch:SSE3")
    SET(AVX_FLAG "/arch:AVX")
    SET(AVX2_FLAG "/arch:AVX2")
    set(AVX512_FLAG "/arch:AVX512f")
    set(VNNI_FLAG "/arch:AVX512vnni")
endif()

set(CMAKE_REQUIRED_FLAGS_RETAINED ${CMAKE_REQUIRED_FLAGS})

# Check  MMX
set(CMAKE_REQUIRED_FLAGS ${MMX_FLAG})
set(MMX_FOUND_EXITCODE 1 CACHE STRING "Result from TRY_RUN" FORCE)
CHECK_CXX_SOURCE_RUNS("
#include <mmintrin.h>
int main()
{
    _mm_setzero_si64();
    return 0;
}" MMX_FOUND)

# Check SSE2
set(CMAKE_REQUIRED_FLAGS ${SSE2_FLAG})
set(SSE2_FOUND_EXITCODE 1 CACHE STRING "Result from TRY_RUN" FORCE)
CHECK_CXX_SOURCE_RUNS("
#include <emmintrin.h>
int main()
{
    _mm_setzero_si128();
    return 0;
}" SSE2_FOUND)

# Check SSE3
set(CMAKE_REQUIRED_FLAGS ${SSE3_FLAG})
set(SSE3_FOUND_EXITCODE 1 CACHE STRING "Result from TRY_RUN" FORCE)
CHECK_CXX_SOURCE_RUNS("
#include <pmmintrin.h>
int main()
{
    __m128d a = _mm_set1_pd(6.28);
    __m128d b = _mm_set1_pd(3.14);
    __m128d result = _mm_addsub_pd(a, b);
    result = _mm_movedup_pd(result);
    return 0;
}" SSE3_FOUND)

# Check AVX
set(CMAKE_REQUIRED_FLAGS ${AVX_FLAG})
set(AVX_FOUND_EXITCODE 1 CACHE STRING "Result from TRY_RUN" FORCE)
CHECK_CXX_SOURCE_RUNS("
#include <immintrin.h>
int main()
{
    __m256 a = _mm256_set_ps (-1.0f, 2.0f, -3.0f, 4.0f, -1.0f, 2.0f, -3.0f, 4.0f);
    __m256 b = _mm256_set_ps (1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f);
    __m256 result = _mm256_add_ps (a, b);
    return 0;
}" AVX_FOUND)

# Check AVX 2
set(CMAKE_REQUIRED_FLAGS ${AVX2_FLAG})
set(AVX2_FOUND_EXITCODE 1 CACHE STRING "Result from TRY_RUN" FORCE)
CHECK_CXX_SOURCE_RUNS("
#include <immintrin.h>
int main()
{
    __m256i a = _mm256_set_epi32 (-1, 2, -3, 4, -1, 2, -3, 4);
    __m256i result = _mm256_abs_epi32 (a);
    return 0;
}" AVX2_FOUND)

# Check AVX512
if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS "5.4")
  # the compiler gcc < 5.4 does not support AVX512, so give True for compiling
  set(AVX512_FOUND ON)
else()
  set(CMAKE_REQUIRED_FLAGS ${AVX512_FLAG})
  set(AVX512_FOUND_EXITCODE 1 CACHE STRING "Result from TRY_RUN" FORCE)
  CHECK_CXX_SOURCE_RUNS("
  #include <immintrin.h>
  int main()
  {
      __m512i a = _mm512_set_epi32 (-1, 2, -3, 4, -1, 2, -3, 4,
                                    13, -5, 6, -7, 9, 2, -6, 3);
      __m512i result = _mm512_abs_epi32 (a);
      return 0;
  }" AVX512_FOUND)
endif()

# Check AVX512 VNNI
set(CMAKE_REQUIRED_FLAGS ${AVX512_FLAG} ${VNNI_FLAG})
set(VNNI_FOUND_EXITCODE 1 CACHE STRING "Result from TRY_RUN" FORCE)
CHECK_CXX_SOURCE_RUNS("
#include <immintrin.h>
int main()
{
    __m512i a = _mm512_set_epi32 (-1, 2, -3, 4, -1, 2, -3, 4,
                                  13, -5, 6, -7, 9, 2, -6, 3);
    __m512i b = _mm512_set_epi32 (-10, 23, -13, 40, -1, 2, -3, 4,
                                  1, -50, 6, -7, 9, 2, -6, 3);
    __m512i c = _mm512_set_epi32 (-1, 2, -3, 4, -1, 2, -3, 4,
                                  3, -1, 6, -7, 9, 2, -6, 3);
    __m512i result = _mm512_dpbusd_epi32(a, b, c);
    return 0;
}" VNNI_FOUND)

set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_RETAINED})
mark_as_advanced(MMX_FOUND SSE2_FOUND SSE3_FOUND AVX_FOUND AVX2_FOUND AVX512_FOUND)
