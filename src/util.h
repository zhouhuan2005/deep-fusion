/*******************************************************************************
 * This file is part of the JITInfer (https://github.com/tensor-tang/jitinfer).
 * Copyright (c) 2018 Tensor Tang.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef WIN32
#include <malloc.h>
#include <windows.h>
#endif

namespace jitinfer {

namespace util {

template <typename T>
inline size_t array_product(const T *p, size_t num) {
  size_t out = 1;
  for (size_t i = 0; i < num; ++i) {
    out *= size_t(p[i]);
  }
  return out;
}

template <typename T, typename P>
inline bool one_of(T val, P item) {
  return val == item;
}
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
  return val == item || one_of(val, item_others...);
}

template <typename T>
struct remove_reference {
  typedef T type;
};
template <typename T>
struct remove_reference<T &> {
  typedef T type;
};
template <typename T>
struct remove_reference<T &&> {
  typedef T type;
};

template <typename T, typename U>
inline typename remove_reference<T>::type div_up(const T a, const U b) {
  assert(b);
  return (a + b - 1) / b;
}
template <typename T>
inline T &&forward(typename remove_reference<T>::type &t) {
  return static_cast<T &&>(t);
}
template <typename T>
inline T &&forward(typename remove_reference<T>::type &&t) {
  return static_cast<T &&>(t);
}

template <typename T>
inline typename remove_reference<T>::type zero() {
  auto zero = typename remove_reference<T>::type();
  return zero;
}

// divide jobs on workers
// for example 4 jobs to 3 worker get 2,1,1
template <typename T, typename U>
inline void balance211(T n, U team, U tid, T &n_start, T &n_end) {
  T n_min = 1;
  T &n_my = n_end;
  if (team <= 1 || n == 0) {
    n_start = 0;
    n_my = n;
  } else if (n_min == 1) {
    // team = T1 + T2
    // n = T1*n1 + T2*n2  (n1 - n2 = 1)
    T n1 = div_up(n, (T)team);
    T n2 = n1 - 1;
    T T1 = n - n2 * (T)team;
    n_my = (T)tid < T1 ? n1 : n2;
    n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
  }

  n_end += n_start;
}

template <typename T>
inline T nd_iterator_init(T start) {
  return start;
}
template <typename T, typename U, typename W, typename... Args>
inline T nd_iterator_init(T start, U &x, const W &X, Args &&... tuple) {
  start = nd_iterator_init(start, forward<Args>(tuple)...);
  x = start % X;
  return start / X;
}

inline bool nd_iterator_step() { return true; }
template <typename U, typename W, typename... Args>
inline bool nd_iterator_step(U &x, const W &X, Args &&... tuple) {
  if (nd_iterator_step(forward<Args>(tuple)...)) {
    x = (x + 1) % X;
    return x == 0;
  }
  return false;
}

namespace timer {
inline double get_current_ms() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
};
}

namespace env {
int _getenv(char *value, const char *name, int length);
bool profiling_time();
bool jit_dump_code();
}
}

void *malloc(size_t size, int alignment = 64);
void free(void *p);
}
