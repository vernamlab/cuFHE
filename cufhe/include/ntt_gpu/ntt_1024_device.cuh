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

#include "ntt_ffp.cuh"
#include "ntt_single_thread.cuh"
#include "ntt_shifting.cuh"
#include <include/details/utils_gpu.cuh>

namespace cufhe {

__constant__ FFP con_twd[1024];
__constant__ FFP con_twd_inv[1024];
__constant__ FFP con_twd_sqrt[1024];
__constant__ FFP con_twd_sqrt_inv[1024];

__device__ inline
void NTT1024Core(FFP* r,
                 FFP* s,
                 const FFP* const twd,
                 const FFP* const twd_sqrt,
                 const uint32_t& t1d,
                 const uint3& t3d) {
  FFP *ptr = nullptr;
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    //r[i] *= twd_sqrt[(i << 7) | t1d]; // mult twiddle sqrt
    r[i] *= con_twd_sqrt[(i << 7) | t1d]; // mult twiddle sqrt
  NTT8(r);
  NTT8x2Lsh(r, t3d.z); // if (t1d >= 64) NTT8x2<1>(r);
  ptr = &s[(t3d.y << 7) | (t3d.z << 6) | (t3d.x << 2)];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[(i >> 2 << 5) | (i & 0x3)] = r[i];
  __syncthreads();

  ptr = &s[(t3d.z << 9) | (t3d.y << 3) | t3d.x];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i << 6];
  NTT2(r);
  NTT2(r + 2);
  NTT2(r + 4);
  NTT2(r + 6);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[i << 6] = r[i];
  __syncthreads();

  ptr = &s[t1d];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    //r[i] = ptr[i << 7] * twd[(i << 7) | t1d]; // mult twiddle
    r[i] = ptr[i << 7] * con_twd[(i << 7) | t1d]; // mult twiddle
  NTT8(r);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[i << 7] = r[i];
  __syncthreads();

  ptr = &s[(t1d >> 2 << 5) | (t3d.x & 0x3)];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i << 2];
  NTT8x8Lsh(r, t1d >> 4); // less divergence if put here!
  NTT8(r);
}

__device__ inline
void NTTInv1024Core(FFP* r,
                    FFP* s,
                    const FFP* const twd_inv,
                    const FFP* const twd_sqrt_inv,
                    const uint32_t& t1d,
                    const uint3& t3d) {

  FFP *ptr = nullptr;
  NTTInv8(r);
  NTTInv8x2Lsh(r, t3d.z); // if (t1d >= 64) NTT8x2<1>(r);
  ptr = &s[(t3d.y << 7) | (t3d.z << 6) | (t3d.x << 2)];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[(i >> 2 << 5) | (i & 0x3)] = r[i];
  __syncthreads();

  ptr = &s[(t3d.z << 9) | (t3d.y << 3) | t3d.x];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i << 6];
  NTT2(r);
  NTT2(r + 2);
  NTT2(r + 4);
  NTT2(r + 6);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[i << 6] = r[i];
  __syncthreads();

  ptr = &s[t1d];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    //r[i] = ptr[i << 7] * twd_inv[(i << 7) | t1d]; // mult twiddle
    r[i] = ptr[i << 7] * con_twd_inv[(i << 7) | t1d]; // mult twiddle
  NTTInv8(r);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    ptr[i << 7] = r[i];
  __syncthreads();

  ptr = &s[(t1d >> 2 << 5) | (t3d.x & 0x3)];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = ptr[i << 2];
  NTTInv8x8Lsh(r, t1d >> 4); // less divergence if put here!
  NTTInv8(r);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    //r[i] *= twd_sqrt_inv[(i << 7) | t1d]; // mult twiddle sqrt
    r[i] *= con_twd_sqrt_inv[(i << 7) | t1d]; // mult twiddle sqrt
}

template <typename T>
__device__
void NTT1024(FFP* out,
             T* in,
             FFP* temp_shared,
             const FFP* const twd,
             const FFP* const twd_sqrt,
             const uint32_t leading_thread) {
  const uint32_t t1d = ThisThreadRankInBlock() - leading_thread;
  uint3 t3d;
  Index3DFrom1D<8, 8, 2>(t3d, t1d);
  register FFP r[8];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = FFP((T)in[(i << 7) | t1d]);
  __syncthreads();
  NTT1024Core(r, temp_shared, twd, twd_sqrt, t1d, t3d);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    out[(i << 7) | t1d] = r[i];
}

template <typename T>
__device__
void NTT1024Decomp(FFP* out,
                   T* in,
                   FFP* temp_shared,
                   FFP* twd,
                   FFP* twd_sqrt,
                   uint32_t rsh_bits,
                   T mask,
                   T offset,
                   const uint32_t leading_thread) {
  const uint32_t t1d = ThisThreadRankInBlock() - leading_thread;
  uint3 t3d;
  Index3DFrom1D<8, 8, 2>(t3d, t1d);
  register FFP r[8];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = FFP(((in[(i << 7) | t1d] >> rsh_bits) & mask) - offset);
  __syncthreads();
  NTT1024Core(r, temp_shared, twd, twd_sqrt, t1d, t3d);
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    out[(i << 7) | t1d] = r[i];
}

template <typename T>
__device__
void NTTInv1024(T* out,
                FFP* in,
                FFP* temp_shared,
                const FFP* const twd_inv,
                const FFP* const twd_sqrt_inv,
                const uint32_t leading_thread) {
  const uint32_t t1d = ThisThreadRankInBlock() - leading_thread;
  uint3 t3d;
  Index3DFrom1D<8, 8, 2>(t3d, t1d);
  register FFP r[8];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = in[(i << 7) | t1d];
  __syncthreads();
  NTTInv1024Core(r, temp_shared, twd_inv, twd_sqrt_inv, t1d, t3d);
  // mod 2^32 specifically
  constexpr uint64_t med = FFP::kModulus() / 2;
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    out[(i << 7) | t1d] = (T)r[i].val() - (r[i].val() > med);
}

template <typename T>
__device__
void NTTInv1024Add(T* out,
                   FFP* in,
                   FFP* temp_shared,
                   const FFP* const twd_inv,
                   const FFP* const twd_sqrt_inv,
                   const uint32_t leading_thread) {
  const uint32_t t1d = ThisThreadRankInBlock() - leading_thread;
  uint3 t3d;
  Index3DFrom1D<8, 8, 2>(t3d, t1d);
  register FFP r[8];
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    r[i] = in[(i << 7) | t1d];
  __syncthreads();
  NTTInv1024Core(r, temp_shared, twd_inv, twd_sqrt_inv, t1d, t3d);
  // mod 2^32 specifically
  constexpr uint64_t med = FFP::kModulus() / 2;
  #pragma unroll
  for (int i = 0; i < 8; i ++)
    out[(i << 7) | t1d] += T(r[i].val() - (r[i].val() >= med));
}

} // namespace cufhe
