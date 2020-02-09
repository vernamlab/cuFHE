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

#include <include/details/allocator_gpu.cuh>
#include "cufhe_core.h"

namespace cufhe {

void BootstrappingKeyToNTT(const BootstrappingKey* bk, int gpuNum);
void KeySwitchingKeyToDevice(const KeySwitchingKey* ksk, int gpuNum);
void DeleteBootstrappingKeyNTT(int gpuNum);
void DeleteKeySwitchingKey(int gpuNum);
void Bootstrap(LWESample* out, LWESample* in, Torus mu, cudaStream_t st,int gpuNum);
void BootstrapTLWE2TRLWE(Torus* out, LWESample* in, Torus mu, cudaStream_t st,int gpuNum);
void NoiselessTrivial(LWESample* out, int p, Torus mu, cudaStream_t st);

void NandBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                   Torus fix, cudaStream_t st, int gpuNum);
void OrBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                 Torus fix, cudaStream_t st, int gpuNum);
void OrYNBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                   Torus fix, cudaStream_t st, int gpuNum);
void OrNYBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                   Torus fix, cudaStream_t st, int gpuNum);
void AndBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                  Torus fix, cudaStream_t st, int gpuNum);
void AndYNBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                    Torus fix, cudaStream_t st, int gpuNum);
void AndNYBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                    Torus fix, cudaStream_t st, int gpuNum);
void NorBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                  Torus fix, cudaStream_t st, int gpuNum);
void XorBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                  Torus fix, cudaStream_t st, int gpuNum);
void XnorBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                   Torus fix, cudaStream_t st, int gpuNum);
void CopyBootstrap(LWESample* out, LWESample* in, cudaStream_t st, int gpuNum);
void NotBootstrap(LWESample* out, LWESample* in, int n, cudaStream_t st, int gpuNum);
void MuxBootstrap(LWESample* out, LWESample* inc, LWESample* in1,
                  LWESample* in0, Torus mu, Torus fix, Torus muxfix,
                  cudaStream_t st, int gpuNum);
}  // namespace cufhe
