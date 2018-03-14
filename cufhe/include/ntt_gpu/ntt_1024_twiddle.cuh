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

#include "ntt_ffp.cuh"
#include "ntt_conv_kind.cuh"

namespace cufhe {

/** All twiddle factors are stored here. */
template <ConvKind conv_kind = NEGATIVE_CYCLIC_CONVOLUTION>
class CuTwiddle {

public:
  __host__ __device__ inline
  CuTwiddle() {
    twd_ = nullptr;
    twd_inv_ = nullptr;
    twd_sqrt_ = nullptr;
    twd_sqrt_inv_ = nullptr;
  }

  __host__ __device__ inline
  ~CuTwiddle() {}

  void Create(uint32_t size = 1024);

  void Destroy();

  FFP* twd_;
  FFP* twd_inv_;
  FFP* twd_sqrt_;
  FFP* twd_sqrt_inv_;
}; // class CuTwiddle

} // namespace cufhe
