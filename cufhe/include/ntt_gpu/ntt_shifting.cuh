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
//// Left-shifting in NTT conversions of size 16, 32, 64 on each thread     ////
////   based on "col" index for the i = 0 ~ 7 rows.                         ////
////////////////////////////////////////////////////////////////////////////////

/** s[i] << 12 * i * col mod P. */
template <uint32_t col>
__device__ inline
void NTT8x2Lsh(FFP* s);

template <>
__device__ inline
void NTT8x2Lsh<0>(FFP* s) {}

template <>
__device__ inline
void NTT8x2Lsh<1>(FFP* s) {
  s[1].Lsh32(12);
  s[2].Lsh32(24);
  s[3].Lsh64(36);
  s[4].Lsh64(48);
  s[5].Lsh64(60);
  s[6].Lsh96(72);
  s[7].Lsh96(84);
}

__device__ inline
void NTT8x2Lsh(FFP* s, uint32_t col) {
  if (1 == col)
    NTT8x2Lsh<1>(s);
}

/** s[i] << 6 * i * col mod P. */
template <uint32_t col>
__device__ inline
void NTT8x4Lsh(FFP* s);

template <>
__device__ inline
void NTT8x4Lsh<0>(FFP* s) {}

template <>
__device__ inline
void NTT8x4Lsh<1>(FFP* s) {
  s[1].Lsh32(6);
  s[2].Lsh32(12);
  s[3].Lsh32(18);
  s[4].Lsh32(24);
  s[5].Lsh32(30);
  s[6].Lsh64(36);
  s[7].Lsh64(42);
}

template <>
__device__ inline
void NTT8x4Lsh<2>(FFP* s) {
  s[1].Lsh32(12);
  s[2].Lsh32(24);
  s[3].Lsh64(36);
  s[4].Lsh64(48);
  s[5].Lsh64(60);
  s[6].Lsh96(72);
  s[7].Lsh96(84);
}

template <>
__device__ inline
void NTT8x4Lsh<3>(FFP* s) {
  s[1].Lsh32(18);
  s[2].Lsh64(36);
  s[3].Lsh64(54);
  s[4].Lsh96(72);
  s[5].Lsh96(90);
  s[6].Lsh128(108);
  s[7].Lsh128(126);
}

__device__ inline
void NTT8x4Lsh(FFP* s, uint32_t col) {
  if (1 == col)
    NTT8x4Lsh<1>(s);
  else if (2 == col)
    NTT8x4Lsh<2>(s);
  else if (3 == col)
    NTT8x4Lsh<3>(s);
}

/** s[i] << 3 * i * col mod P. */

template <uint32_t col>
__device__ inline
void NTT8x8Lsh(FFP* s);

template <>
__device__ inline
void NTT8x8Lsh<0>(FFP* s) {}

template <>
__device__ inline
void NTT8x8Lsh<1>(FFP* s) {
  s[1].Lsh32(3);
  s[2].Lsh32(6);
  s[3].Lsh32(9);
  s[4].Lsh32(12);
  s[5].Lsh32(15);
  s[6].Lsh32(18);
  s[7].Lsh32(21);
}

template <>
__device__ inline
void NTT8x8Lsh<2>(FFP* s) {
  s[1].Lsh32(6);
  s[2].Lsh32(12);
  s[3].Lsh32(18);
  s[4].Lsh32(24);
  s[5].Lsh32(30);
  s[6].Lsh64(36);
  s[7].Lsh64(42);
}

template <>
__device__ inline
void NTT8x8Lsh<3>(FFP* s) {
  s[1].Lsh32(9);
  s[2].Lsh32(18);
  s[3].Lsh32(27);
  s[4].Lsh64(36);
  s[5].Lsh64(45);
  s[6].Lsh64(54);
  s[7].Lsh64(63);
}

template <>
__device__ inline
void NTT8x8Lsh<4>(FFP* s) {
  s[1].Lsh32(12);
  s[2].Lsh32(24);
  s[3].Lsh64(36);
  s[4].Lsh64(48);
  s[5].Lsh64(60);
  s[6].Lsh96(72);
  s[7].Lsh96(84);
}

template <>
__device__ inline
void NTT8x8Lsh<5>(FFP* s) {
  s[1].Lsh32(15);
  s[2].Lsh32(30);
  s[3].Lsh64(45);
  s[4].Lsh64(60);
  s[5].Lsh96(75);
  s[6].Lsh96(90);
  s[7].Lsh128(105);
}

template <>
__device__ inline
void NTT8x8Lsh<6>(FFP* s) {
  s[1].Lsh32(18);
  s[2].Lsh64(36);
  s[3].Lsh64(54);
  s[4].Lsh96(72);
  s[5].Lsh96(90);
  s[6].Lsh128(108);
  s[7].Lsh128(126);
}

template <>
__device__ inline
void NTT8x8Lsh<7>(FFP* s) {
  s[1].Lsh32(21);
  s[2].Lsh64(42);
  s[3].Lsh64(63);
  s[4].Lsh96(84);
  s[5].Lsh128(105);
  s[6].Lsh128(126);
  s[7].Lsh160(147);
}

__device__ inline
void NTT8x8Lsh(FFP* s, uint32_t col) {
  if (1 == col)
    NTT8x8Lsh<1>(s);
  else if (2 == col)
    NTT8x8Lsh<2>(s);
  else if (3 == col)
    NTT8x8Lsh<3>(s);
  else if (4 == col)
    NTT8x8Lsh<4>(s);
  else if (5 == col)
    NTT8x8Lsh<5>(s);
  else if (6 == col)
    NTT8x8Lsh<6>(s);
  else if (7 == col)
    NTT8x8Lsh<7>(s);
}

/** s[i] << 192 - 12 * i * col mod P. */
template <uint32_t col>
__device__ inline
void NTTInv8x2Lsh(FFP* s);

template <>
__device__ inline
void NTTInv8x2Lsh<0>(FFP* s) {}

template <>
__device__ inline
void NTTInv8x2Lsh<1>(FFP* s) {
  s[1].Lsh192(180);
  s[2].Lsh192(168);
  s[3].Lsh160(156);
  s[4].Lsh160(144);
  s[5].Lsh160(132);
  s[6].Lsh128(120);
  s[7].Lsh128(108);
}

__device__ inline
void NTTInv8x2Lsh(FFP* s, uint32_t col) {
  if (1 == col)
    NTTInv8x2Lsh<1>(s);
}

/** s[i] << 192 - 6 * i * col mod P. */
template <uint32_t col>
__device__ inline
void NTTInv8x4Lsh(FFP* s);

template <>
__device__ inline
void NTTInv8x4Lsh<0>(FFP* s) {}

template <>
__device__ inline
void NTTInv8x4Lsh<1>(FFP* s) {
  s[1].Lsh192(186);
  s[2].Lsh192(180);
  s[3].Lsh192(174);
  s[4].Lsh192(168);
  s[5].Lsh192(162);
  s[6].Lsh160(156);
  s[7].Lsh160(150);
}

template <>
__device__ inline
void NTTInv8x4Lsh<2>(FFP* s) {
  s[1].Lsh192(180);
  s[2].Lsh192(168);
  s[3].Lsh160(156);
  s[4].Lsh160(144);
  s[5].Lsh160(132);
  s[6].Lsh128(120);
  s[7].Lsh128(108);
}

template <>
__device__ inline
void NTTInv8x4Lsh<3>(FFP* s) {
  s[1].Lsh192(174);
  s[2].Lsh160(156);
  s[3].Lsh160(138);
  s[4].Lsh128(120);
  s[5].Lsh128(102);
  s[6].Lsh96(84);
  s[7].Lsh96(66);
}

__device__ inline
void NTTInv8x4Lsh(FFP* s, uint32_t col) {
  if (1 == col)
    NTTInv8x4Lsh<1>(s);
  else if (2 == col)
    NTTInv8x4Lsh<2>(s);
  else if (3 == col)
    NTTInv8x4Lsh<3>(s);
}

/** s[i] << 192 - 6 * i * col mod P. */
template <uint32_t col>
__device__ inline
void NTTInv8x8Lsh(FFP* s);

template <>
__device__ inline
void NTTInv8x8Lsh<0>(FFP* s) {}

template <>
__device__ inline
void NTTInv8x8Lsh<1>(FFP* s) {
  s[1].Lsh192(189);
  s[2].Lsh192(186);
  s[3].Lsh192(183);
  s[4].Lsh192(180);
  s[5].Lsh192(177);
  s[6].Lsh192(174);
  s[7].Lsh192(171);
}

template <>
__device__ inline
void NTTInv8x8Lsh<2>(FFP* s) {
  s[1].Lsh192(186);
  s[2].Lsh192(180);
  s[3].Lsh192(174);
  s[4].Lsh192(168);
  s[5].Lsh192(162);
  s[6].Lsh160(156);
  s[7].Lsh160(150);
}

template <>
__device__ inline
void NTTInv8x8Lsh<3>(FFP* s) {
  s[1].Lsh192(183);
  s[2].Lsh192(174);
  s[3].Lsh192(165);
  s[4].Lsh160(156);
  s[5].Lsh160(147);
  s[6].Lsh160(138);
  s[7].Lsh160(129);
}

template <>
__device__ inline
void NTTInv8x8Lsh<4>(FFP* s) {
  s[1].Lsh192(180);
  s[2].Lsh192(168);
  s[3].Lsh160(156);
  s[4].Lsh160(144);
  s[5].Lsh160(132);
  s[6].Lsh128(120);
  s[7].Lsh128(108);
}

template <>
__device__ inline
void NTTInv8x8Lsh<5>(FFP* s) {
  s[1].Lsh192(177);
  s[2].Lsh192(162);
  s[3].Lsh160(147);
  s[4].Lsh160(132);
  s[5].Lsh128(117);
  s[6].Lsh128(102);
  s[7].Lsh96(87);
}

template <>
__device__ inline
void NTTInv8x8Lsh<6>(FFP* s) {
  s[1].Lsh192(174);
  s[2].Lsh160(156);
  s[3].Lsh160(138);
  s[4].Lsh128(120);
  s[5].Lsh128(102);
  s[6].Lsh96(84);
  s[7].Lsh96(66);
}

template <>
__device__ inline
void NTTInv8x8Lsh<7>(FFP* s) {
  s[1].Lsh192(171);
  s[2].Lsh160(150);
  s[3].Lsh160(129);
  s[4].Lsh128(108);
  s[5].Lsh96(87);
  s[6].Lsh96(66);
  s[7].Lsh64(45);
}

__device__ inline
void NTTInv8x8Lsh(FFP* s, uint32_t col) {
  if (1 == col)
    NTTInv8x8Lsh<1>(s);
  else if (2 == col)
    NTTInv8x8Lsh<2>(s);
  else if (3 == col)
    NTTInv8x8Lsh<3>(s);
  else if (4 == col)
    NTTInv8x8Lsh<4>(s);
  else if (5 == col)
    NTTInv8x8Lsh<5>(s);
  else if (6 == col)
    NTTInv8x8Lsh<6>(s);
  else if (7 == col)
    NTTInv8x8Lsh<7>(s);
}

} // namespace cufhe
