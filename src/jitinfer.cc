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
#include "jitinfer.h"
#include "jitinfer_common.h"
#include "util.h"

#include "op_concat.h"

namespace jitinfer {

memory::dims nchw2format(const memory::nchw_dims& dm,
                         const memory::format fmt) {
  using format = memory::format;
  memory::dims out;
  switch (fmt) {
    case format::nhwc:
      out.resize(4);
      out[0] = dm[0];
      out[1] = dm[2];
      out[2] = dm[3];
      out[3] = dm[1];
      break;
    case format::nchw:
      out.resize(4);
      out[0] = dm[0];
      out[1] = dm[1];
      out[2] = dm[2];
      out[3] = dm[3];
      break;
    default:
      // log error, exit
      assert(!"bad type");
  }

  check_eq(util::array_product<int>(dm.data(), dm.size()),
           util::array_product<int>(out.data(), out.size()));
  return out;
}

memory::memory(const nchw_dims& dm,
               const format fmt,
               const dtype dt,
               int alignment)
    : fmt_(fmt), dt_(dt) {
  dims_ = nchw2format(dm, fmt);
  allocate_buffer(alignment);
}

memory::~memory() { free(data_); }

void memory::allocate_buffer(int alignment) {
  assert(buffer_size() > 0);
  data_ = malloc(buffer_size(), alignment);
  assert(data_ != NULL);
}

size_t memory::size() {
  return util::array_product<int>(dims_.data(), dims_.size());
}

size_t memory::buffer_size() { return size() * dtype_size(dt_); }

void op::submit() {
  // TODO: add timer
  infer();
}
std::unique_ptr<op> concat(const std::vector<std::unique_ptr<memory>>& srcs,
                           std::unique_ptr<memory>& dst,
                           bool post_relu) {
  switch (dst->data_type()) {
#define CASE(tp)          \
  case memory::dtype::tp: \
    return std::unique_ptr<op>(new op_concat<tp>(srcs, dst, post_relu))
    CASE(f32);
    CASE(s32);
    CASE(s8);
    CASE(u8);
#undef CASE
    default:
      assert(!"bad data_type");
  }
  return nullptr;
}
}
