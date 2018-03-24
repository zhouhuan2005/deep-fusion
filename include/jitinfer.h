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

#include <stdlib.h>
#include <array>
#include <memory>
#include <vector>

namespace jitinfer {

// Disable the copy and assignment operator for a class.
#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(classname)          \
private:                                            \
  classname(const classname &) = delete;            \
  classname(const classname &&) = delete;           \
  classname &operator=(const classname &) = delete; \
  classname &operator=(const classname &&) = delete
#endif

struct opdesc {
  int tmp;
};

struct memory {
public:
  enum format {
    format_undef = 0,
    x,
    nchw,
    nhwc,
  };
  typedef std::vector<int> dims;
  typedef std::array<int, 4> nchw_dims;

  enum dtype {
    f32 = 0,
    s32,
    s8,
    u8,
  };

  // TODO: enable more format init
  // explicit memory(const dims& dm, const format fmt, const dtype dt, int
  // alignment = 64);
  explicit memory(const nchw_dims &dm,
                  const format fmt,
                  const dtype dt,
                  int alignment = 64);
  ~memory();
  size_t size();
  size_t buffer_size();
  dims actual_dims() { return dims_; }
  void *data() { return data_; }

private:
  void allocate_buffer(int alignment);
  void *data_;
  dims dims_;
  format fmt_;
  dtype dt_;

  DISABLE_COPY_AND_ASSIGN(memory);
};

class op {
public:
  explicit op() {}
  virtual void execute();

protected:
  virtual void infer() = 0;
  DISABLE_COPY_AND_ASSIGN(op);
};

std::unique_ptr<op> concat(const std::vector<memory> &srcs,
                           memory dst,
                           bool post_relu = false);
}
