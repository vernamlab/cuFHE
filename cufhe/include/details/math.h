/**
 * Copyright 2018 Wei Dai <wdai3141@gmail.com>
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

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

namespace cufhe {

__host__ __device__ inline
uint32_t Pow2(uint32_t e) { return (uint32_t)0x1 << e; }

__host__ __device__ inline
uint32_t Log2(uint32_t x) { return x < 2 ? 0 : 1 + Log2(x / 2); }

__host__ __device__ inline
constexpr uint32_t Pow2Const(uint32_t e) { return (uint32_t)0x1 << e; }

__host__ __device__ inline
constexpr uint32_t Log2Const(uint32_t x) {
  return x < 2 ? 0 : 1 + Log2Const(x / 2);
}

__host__ __device__ inline
uint32_t NextPow2(uint32_t x) { return Pow2(Log2(x - 1) + 1); }

__host__ __device__ inline
constexpr uint32_t NextPow2Const(uint32_t x) {
  return Pow2Const(Log2Const(x - 1) + 1);
}

__host__ __device__ inline
uint32_t Align(uint32_t x, uint32_t width) {
  return (x + width - 1) / width * width;
}

__host__ __device__ inline
constexpr uint32_t AlignConst(uint32_t x, uint32_t width) {
  return (x + width - 1) / width * width;
}

__host__ __device__ inline
uint32_t Align512(uint32_t x) { return (x + 511) >> 9 << 9; }

__host__ __device__ inline
constexpr uint32_t Align512Const(uint32_t x) { return (x + 511) >> 9 << 9; }

} // namespace cufhe
