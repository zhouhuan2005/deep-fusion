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
/**
 * This file defines some utilities of MKL-DNN, or exchange with jitinfer
 */
#pragma once

#include "jitinfer.h"
#include "mkldnn.hpp"

namespace jitinfer {
namespace util {

std::unique_ptr<mkldnn::eltwise_forward::primitive_desc> get_mkldnn_relu_pd(
    const mkldnn::memory::desc md, const mkldnn::engine& eng);

// exchange btw mkldnn and jitinfer
namespace exchange {
mkldnn::memory::dims dims(const memory::nchw_dims& nchwdims);
memory::dtype dtype(mkldnn::memory::data_type dt);
mkldnn::memory::data_type dtype(memory::dtype dt);
}
}
}
