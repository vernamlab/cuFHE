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

namespace cufhe {

////////////////////////////////////////////////////////////////////////////////
//// NTT conversions of sizes 2, 4, 8 on a single thread over registers     ////
////////////////////////////////////////////////////////////////////////////////

__device__ inline
void NTT2(FFP& r0, FFP& r1) {
  register FFP t = r0 - r1;
  r0 += r1;
  r1 = t;
}

__device__ inline
void NTT2(FFP* r) {
  NTT2(r[0], r[1]);
}

__device__ inline
void NTTInv2(FFP& r0, FFP& r1) {
  NTT2(r0, r1);
}

__device__ inline
void NTTInv2(FFP* r) {
  NTT2(r);
}

__device__ inline
void NTT4(FFP& r0, FFP& r1, FFP& r2, FFP& r3) {
  NTT2(r0, r2);
  NTT2(r1, r3);
  r3.Lsh64(48);
  NTT2(r0, r1);
  NTT2(r2, r3);
  r1.Swap(r2);
}

__device__ inline
void NTT4(FFP* r) {
  NTT4(r[0], r[1], r[2], r[3]);
}

__device__ inline
void NTTInv4(FFP& r0, FFP& r1, FFP& r2, FFP& r3) {
  NTTInv2(r0, r2);
  NTTInv2(r1, r3);
  r3.Lsh160(144);
  NTTInv2(r0, r1);
  NTTInv2(r2, r3);
  r1.Swap(r2);
}

__device__ inline
void NTTInv4(FFP* r) {
  NTTInv4(r[0], r[1], r[2], r[3]);
}

__device__ inline
void NTT8(FFP* r) {
  NTT2(r[0], r[4]);
  NTT2(r[1], r[5]);
  NTT2(r[2], r[6]);
  NTT2(r[3], r[7]);
  r[5].Lsh32(24);
  r[6].Lsh64(48);
  r[7].Lsh96(72);
  // instead of calling NTT4 ...
  NTT2(r[0], r[2]);
  NTT2(r[1], r[3]);
  r[3].Lsh64(48);
  NTT2(r[4], r[6]);
  NTT2(r[5], r[7]);
  r[7].Lsh64(48);
  NTT2(r);
  NTT2(&r[2]);
  NTT2(&r[4]);
  NTT2(&r[6]);
  // ... we save 2 swaps (otherwise 4) here
  r[1].Swap(r[4]);
  r[3].Swap(r[6]);
}

__device__ inline
void NTTInv8(FFP* r) {
  NTTInv2(r[0], r[4]);
  NTTInv2(r[1], r[5]);
  NTTInv2(r[2], r[6]);
  NTTInv2(r[3], r[7]);
  r[5].Lsh192(168);
  r[6].Lsh160(144);
  r[7].Lsh128(120);
  // instead of calling NTT4 ...
  NTTInv2(r[0], r[2]);
  NTTInv2(r[1], r[3]);
  r[3].Lsh160(144);
  NTTInv2(r[4], r[6]);
  NTTInv2(r[5], r[7]);
  r[7].Lsh160(144);
  NTTInv2(r);
  NTTInv2(&r[2]);
  NTTInv2(&r[4]);
  NTTInv2(&r[6]);
  // ... we save 2 swaps (otherwise 4) here
  r[1].Swap(r[4]);
  r[3].Swap(r[6]);
}

} // namespace cufhe
