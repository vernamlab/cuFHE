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

#include <include/bootstrap_cpu.h>
#include <iostream>
#include <string.h>
using namespace std;

namespace cufhe {

uint32_t ModSwitchFromTorus(Torus phase, uint32_t space){
  uint64_t interv = ((UINT64_C(1) << 63) / space) * 2;
  uint64_t half_interval = interv/2;
  uint64_t phase64 = (uint64_t(phase) << 32) + half_interval;
  return phase64 / interv;
}

void PolyMulPowX(Torus* out, Torus* in, uint32_t exp, uint32_t n) {
  Torus* temp = new Torus[n];
  if (exp < n) {
    for (int i = 0; i < exp; i ++)
      temp[i] = -in[i - exp + n];
    for (int i = exp; i < n; i ++)
      temp[i] = in[i - exp];
  }
  else {
    exp -= n;
    for (int i = 0; i < exp; i ++)
      temp[i] = in[i - exp + n];
    for (int i = exp; i < n; i ++)
      temp[i] = -in[i - exp];
  }
  for (int i = 0; i < n; i ++)
    out[i] = temp[i];
  delete [] temp;
}

void PolyMulAdd(Torus* out, Torus* in0, int32_t* in1, uint32_t n) {
  for (int i = 0; i < n; i ++) {
    for (int j = 0; j <= i; j ++)
      out[i] += in0[j] * in1[i - j];
    for (int j = i + 1; j < n; j ++)
      out[i] -= in0[j] * in1[n + i - j];
  }
}

void PolySub(Torus* out, Torus* in0, Torus* in1, uint32_t n) {
  for (int i = 0; i < n; i ++)
    out[i] = in0[i] - in1[i];
}

void PolyDecomp(int32_t** out, Torus* in, uint32_t n,
                uint32_t bits, uint32_t l,
                uint32_t mask, uint32_t half, uint32_t offset) {
  for (int j = 0; j < n; j ++)
    for(int i = 0; i < l; i ++)
      out[i][j] = (((in[j] + offset) >> (32 - (i + 1) * bits)) & mask) - half;
}

void LWESampleSub(LWESample* out, LWESample* in0, LWESample* in1) {
  for(int i = 0; i <= out->n(); i ++)
    out->data()[i] = in0->data()[i] - in1->data()[i];
}

void Bootstrap(LWESample* out,
               LWESample* in,
               Torus mu,
               BootstrappingKey* bk,
               KeySwitchingKey* ksk) {
  uint32_t lwe_n = ksk->n();
  uint32_t tlwe_n = bk->n();
  uint32_t n2 = 2 * tlwe_n;
  uint32_t k = bk->k();
  uint32_t bk_l = bk->l();
  uint32_t bk_bits = bk->w();
  uint32_t bk_mask = (1 << bk_bits) - 1;
  uint32_t bk_half = 1 << (bk_bits - 1);
  uint32_t bk_offset = 0;
  for(int i = 0; i < bk_l; i ++)
    bk_offset += 0x1 << (32 - (i + 1) * bk_bits);
  bk_offset *= bk_half;
  uint32_t kpl = (k + 1) * bk_l;
  uint32_t ksk_l = ksk->l();
  uint32_t ksk_bits = ksk->w();
  uint32_t ksk_mask = (0x1 << ksk_bits) - 1;
  uint32_t ksk_offset = 0x1 << (31 - ksk_l * ksk_bits);
  Torus* temp = new Torus[tlwe_n];
  TLWESample* acc = new TLWESample(bk->n(), bk->k());
  std::pair<void*, MemoryDeleter> pair = AllocatorCPU::New(acc->SizeMalloc());
  acc->set_data((TLWESample::PointerType)pair.first);
  MemoryDeleter acc_deleter = pair.second;
  int** decomp = new int32_t*[kpl];
  for (int i = 0; i < kpl; i ++)
    decomp[i] = new int32_t[tlwe_n];

  Torus bar_b = ModSwitchFromTorus(in->b(), n2);
  for (int i = 0; i < tlwe_n; i ++)
    temp[i] = mu;
  memset(acc->data(), 0, acc->SizeData());
  PolyMulPowX(acc->b(), temp, n2 - bar_b, tlwe_n);

  Torus bar_a;
  for (int i = 0; i < lwe_n; i ++) {
    bar_a = ModSwitchFromTorus(in->a()[i], n2);
    for (int j = 0; j <= k; j ++) {
      PolyMulPowX(temp, acc->ExtractPoly(j), bar_a, tlwe_n);
      PolySub(temp, temp, acc->ExtractPoly(j), tlwe_n);
      PolyDecomp(decomp + j * bk_l, temp, tlwe_n, bk_bits, bk_l,
                 bk_mask, bk_half, bk_offset);
    }
    for (int j = 0; j <= k; j ++)
      for (int p = 0; p < kpl; p ++)
        PolyMulAdd(acc->ExtractPoly(j),
                   bk->ExtractTGSWSample(i).ExtractTLWESample(p).ExtractPoly(j),
                   decomp[p], tlwe_n);

  }

  uint32_t coeff, digit;
  memset(out->data(), 0, out->SizeData());
  out->b() = acc->b()[0];
  LWESample* ksk_entry = new LWESample(lwe_n);
  for (int i = 0; i < k * tlwe_n; i ++) {
    if (i == 0)
      coeff = acc->a()[i];
    else
      coeff = -acc->a()[k * tlwe_n - i];
    coeff += ksk_offset;
    for (int j = 0; j < ksk_l; j ++) {
      digit = (coeff >> (32 - (j + 1) * ksk_bits)) & ksk_mask;
      if (digit != 0) {
        ksk->ExtractLWESample(ksk_entry, ksk->GetLWESampleIndex(i, j, digit));
        LWESampleSub(out, out, ksk_entry);
      }
    }
  }

  for (int i = 0; i < kpl; i ++)
    delete [] decomp[i];
  delete [] decomp;
  delete [] temp;
  acc_deleter(acc->data());
  delete acc;
}

} // namespace cufhe
