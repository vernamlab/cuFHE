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

#include <include/ntt_gpu/ntt_1024_twiddle.cuh>
#include <include/details/error_gpu.cuh>
#include <include/details/assert.h>
#include <include/details/allocator_gpu.cuh>

namespace cufhe {

__global__
void __GenTwd__(FFP* twd, FFP* twd_inv) {
  uint32_t n = 1024;
  uint32_t idx;
  uint32_t cid;
  FFP w = FFP::Root(n);
  FFP t;
  uint32_t e;
  cid = (threadIdx.z << 6) + (threadIdx.y << 3) + threadIdx.x;
  for (int i = 0; i < 8; i ++) {
    e = (threadIdx.z * 8 + threadIdx.y / 4 * 4 + (threadIdx.x % 4))
      * (i * 8 + (threadIdx.y % 4) * 2 + threadIdx.x / 4);
    idx = (i * n / 8) + cid;
    twd[idx] = FFP::Pow(w, e);
    twd_inv[idx] = FFP::Pow(w, (n - e) % n);
  }
}

__global__
void __GenTwdSqrt__(FFP* twd_sqrt, FFP* twd_sqrt_inv) {
  uint32_t n = 1024;
  uint32_t idx = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
  FFP w = FFP::Root(2 * n);
  FFP n_inv = FFP::InvPow2(10);
  twd_sqrt[idx] = FFP::Pow(w, idx);
  twd_sqrt_inv[idx] = FFP::Pow(w, (2 * n - idx) % (2 * n)) * n_inv;
}

template <>
void CuTwiddle<NEGATIVE_CYCLIC_CONVOLUTION>::Create(uint32_t size) {
  assert(this->twd_ == nullptr);
  size_t nbytes = sizeof(FFP) * 1024 * 4;
  this->twd_ = (FFP*)AllocatorGPU::New(nbytes).first;
  this->twd_inv_ = this->twd_ + 1024;
  this->twd_sqrt_ = this->twd_inv_ + 1024;
  this->twd_sqrt_inv_ = this->twd_sqrt_ + 1024;
  __GenTwd__<<<1, dim3(8, 8, 2)>>>(this->twd_, this->twd_inv_);
  __GenTwdSqrt__<<<16, 64>>>(this->twd_sqrt_, this->twd_sqrt_inv_);
  cudaDeviceSynchronize();
  CuCheckError();
}

template <>
void CuTwiddle<NEGATIVE_CYCLIC_CONVOLUTION>::Destroy() {
  assert(this->twd_ != nullptr);
  CuSafeCall(cudaFree(this->twd_));
  this->twd_ = nullptr;
  this->twd_inv_ = nullptr;
  this->twd_sqrt_ = nullptr;
  this->twd_sqrt_inv_ = nullptr;
}

} // namespace cufhe
