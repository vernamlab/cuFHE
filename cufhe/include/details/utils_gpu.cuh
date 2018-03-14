/**
 * Copyright 2013-2017 Wei Dai <wdai3141@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <stdint.h>

namespace cufhe {

__device__ inline
uint32_t ThisBlockRankInGrid() {
  return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}

__device__ inline
uint32_t ThisGridSize() {
  return gridDim.x * gridDim.y * gridDim.z;
}

__device__ inline
uint32_t ThisThreadRankInBlock() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

__device__ inline
uint32_t ThisBlockSize() {
  return blockDim.x * blockDim.y * blockDim.z;
}

template <uint32_t dim_x, uint32_t dim_y, uint32_t dim_z>
__device__ inline
void Index3DFrom1D(uint3& t3d, uint32_t t1d) {
  t3d.x = t1d % dim_x;
  t1d /= dim_x;
  t3d.y = t1d % dim_y;
  t3d.z = t1d / dim_y;
}

template <uint32_t dim_x, uint32_t dim_y, uint32_t dim_z>
__device__ inline
uint3 Index1DFrom3D(uint32_t& t1d, uint3 t3d) {
  t1d = t3d.x + dim_x * (t3d.y + dim_y * t3d.z);
}

} // namespace cufhe
