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

#include "jitinfer.h"
#include "mkldnn.hpp"

namespace jitinfer {

memory::dtype mkldnn2jitinfer(mkldnn::memory::data_type dt) {
  switch (dt) {
#define CASE(tp)                      \
  case mkldnn::memory::data_type::tp: \
    return memory::dtype::tp
    CASE(f32);
    CASE(s32);
    CASE(s8);
    CASE(u8);
#undef CASE
    default:
      assert(!"bad data_type");
  }
}

mkldnn::memory::data_type jitinfer2mkldnn(memory::dtype dt) {
  switch (dt) {
#define CASE(tp)          \
  case memory::dtype::tp: \
    return mkldnn::memory::data_type::tp
    CASE(f32);
    CASE(s32);
    CASE(s8);
    CASE(u8);
#undef CASE
    default:
      assert(!"bad data_type");
  }
}
}
