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

#include <include/cufhe.h>
#include <stdio.h>
#include <unistd.h>
#include <include/bootstrap_gpu.cuh>
#include <include/details/error_gpu.cuh>
#include <include/ntt_gpu/ntt.cuh>

#include <iostream>
#include <vector>
using namespace std;

namespace cufhe {

using BootstrappingKeyNTT = TGSWSampleArray_T<FFP>;
// BootstrappingKeyNTT* bk_ntt = nullptr;
// MemoryDeleter bk_ntt_deleter = nullptr;
// KeySwitchingKey* ksk_dev = nullptr;
// MemoryDeleter ksk_dev_deleter = nullptr;
// CuNTTHandler<>* ntt_handler = nullptr;

vector<BootstrappingKeyNTT*> bk_ntts;
vector<MemoryDeleter> bk_ntt_deleters;
vector<CuNTTHandler<>*> ntt_handlers;
vector<KeySwitchingKey*> ksk_devs;
vector<MemoryDeleter> ksk_dev_deleters;

__global__ void __BootstrappingKeyToNTT__(BootstrappingKeyNTT bk_ntt,
                                          BootstrappingKey bk,
                                          CuNTTHandler<> ntt)
{
    __shared__ FFP sh_temp[1024];

    TGSWSample tgsw;
    bk.ExtractTGSWSample(&tgsw, blockIdx.z);
    TLWESample tlwe;
    tgsw.ExtractTLWESample(&tlwe, blockIdx.y);
    Torus* poly_in = tlwe.ExtractPoly(blockIdx.x);

    TGSWSample_T<FFP> tgsw_ntt;
    bk_ntt.ExtractTGSWSample(&tgsw_ntt, blockIdx.z);
    TLWESample_T<FFP> tlwe_ntt;
    tgsw_ntt.ExtractTLWESample(&tlwe_ntt, blockIdx.y);
    FFP* poly_out = tlwe_ntt.ExtractPoly(blockIdx.x);
    ntt.NTT<Torus>(poly_out, poly_in, sh_temp, 0);
}

void BootstrappingKeyToNTT(const BootstrappingKey* bk, int gpuNum)
{
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        BootstrappingKey* d_bk;
        d_bk =
            new BootstrappingKey(bk->n(), bk->k(), bk->l(), bk->w(), bk->t());
        std::pair<void*, MemoryDeleter> pair;

        // Allocate GPU Memory
        pair = AllocatorGPU::New(d_bk->SizeMalloc());
        d_bk->set_data((BootstrappingKey::PointerType)pair.first);
        MemoryDeleter d_bk_deleter = pair.second;
        CuSafeCall(cudaMemcpy(d_bk->data(), bk->data(), d_bk->SizeMalloc(),
                              cudaMemcpyHostToDevice));
        Assert(bk_ntts.size() == i);

        bk_ntts.push_back(new BootstrappingKeyNTT(bk->n(), bk->k(), bk->l(),
                                                  bk->w(), bk->t()));

        // Allocate GPU Memory
        pair = AllocatorGPU::New(bk_ntts[i]->SizeMalloc());
        bk_ntts[i]->set_data((BootstrappingKeyNTT::PointerType)pair.first);
        bk_ntt_deleters.push_back(pair.second);

        Assert(ntt_handlers.size() == i);
        ntt_handlers.push_back(new CuNTTHandler<>());
        ntt_handlers[i]->Create();
        ntt_handlers[i]->CreateConstant();
        cudaDeviceSynchronize();
        CuCheckError();

        dim3 grid(bk->k() + 1, (bk->k() + 1) * bk->l(), bk->t());
        dim3 block(128);
        __BootstrappingKeyToNTT__<<<grid, block>>>(*bk_ntts[i], *d_bk,
                                                   *ntt_handlers[i]);
        cudaDeviceSynchronize();
        CuCheckError();

        d_bk_deleter(d_bk->data());
        delete d_bk;
    }
}

void DeleteBootstrappingKeyNTT(int gpuNum)
{
    for (int i = 0; i < bk_ntts.size(); i++) {
        cudaSetDevice(i);
        bk_ntt_deleters[i](bk_ntts[i]->data());
        delete bk_ntts[i];
        bk_ntts[i] = nullptr;

        ntt_handlers[i]->Destroy();
        delete ntt_handlers[i];
    }
    bk_ntts.clear();
    bk_ntt_deleters.clear();
    ntt_handlers.clear();
}

void KeySwitchingKeyToDevice(const KeySwitchingKey* ksk, int gpuNum)
{
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        Assert(ksk_devs.size() == i);
        ksk_devs.push_back(
            new KeySwitchingKey(ksk->n(), ksk->l(), ksk->w(), ksk->m()));

        std::pair<void*, MemoryDeleter> pair;
        pair = AllocatorGPU::New(ksk_devs[i]->SizeMalloc());
        ksk_devs[i]->set_data((KeySwitchingKey::PointerType)pair.first);
        ksk_dev_deleters.push_back(pair.second);
        CuSafeCall(cudaMemcpy(ksk_devs[i]->data(), ksk->data(),
                              ksk->SizeMalloc(), cudaMemcpyHostToDevice));
    }
}

void DeleteKeySwitchingKey(int gpuNum)
{
    for (int i = 0; i < ksk_devs.size(); i++) {
        cudaSetDevice(i);
        ksk_dev_deleters[i](ksk_devs[i]->data());
        delete ksk_devs[i];
        ksk_devs[i] = nullptr;
    }
    ksk_dev_deleters.clear();
    ksk_devs.clear();
}

__device__ inline uint32_t ModSwitch2048(uint32_t a)
{
    return (((uint64_t)a << 32) + (0x1UL << 52)) >> 53;
}

template <uint32_t lwe_n = 500, uint32_t tlwe_n = 1024,
          uint32_t decomp_bits = 2, uint32_t decomp_size = 8>
__device__ inline void KeySwitch(Torus* lwe, Torus* tlwe, Torus* ksk)
{
    static const Torus decomp_mask = (1u << decomp_bits) - 1;
    static const Torus decomp_offset = 1u << (31 - decomp_size * decomp_bits);
    uint32_t tid = ThisThreadRankInBlock();
    uint32_t bdim = ThisBlockSize();
    Torus tmp;
    Torus res = 0;
    Torus val = 0;
#pragma unroll 0
    for (int i = tid; i <= lwe_n; i += bdim) {
        if (i == lwe_n) res = tlwe[tlwe_n];
#pragma unroll 0
        for (int j = 0; j < tlwe_n; j++) {
            if (j == 0)
                tmp = tlwe[0];
            else
                tmp = -tlwe[1024 - j];
            tmp += decomp_offset;
            for (int k = 0; k < decomp_size; k++) {
                val = (tmp >> (32 - (k + 1) * decomp_bits)) & decomp_mask;
                if (val != 0)
                    res -= ksk[(j << 14) | (k << 11) | (val << 9) | i];
            }
        }
        lwe[i] = res;
    }
}

template <uint32_t lwe_n = 500, uint32_t tlwe_n = 1024, uint32_t tlwe_nbit = 10>
__device__ inline void RotatedTestVector(Torus* tlwe, int32_t bar, uint32_t mu)
{
    // volatile is needed to make register usage of Mux to 128.
    // Reference
    // https://devtalk.nvidia.com/default/topic/466758/cuda-programming-and-performance/tricks-to-fight-register-pressure-or-how-i-got-down-from-29-to-15-registers-/
    volatile uint32_t tid = ThisThreadRankInBlock();
    volatile uint32_t bdim = ThisBlockSize();
    uint32_t cmp, neg, pos;
#pragma unroll
    for (int i = tid; i < tlwe_n; i += bdim) {
        tlwe[i] = 0;  // part a
        if (bar == 2 * tlwe_n)
            tlwe[i + tlwe_n] = mu;
        else {
            cmp = (uint32_t)(i < (bar & 1023));
            neg = -(cmp ^ (bar >> tlwe_nbit));
            pos = -((1 - cmp) ^ (bar >> tlwe_nbit));
            tlwe[i + tlwe_n] = (mu & pos) + ((-mu) & neg);  // part b
        }
    }
    __syncthreads();
}

__device__ void Accumulate(Torus* tlwe, FFP* sh_acc_ntt, FFP* sh_res_ntt,
                           uint32_t a_bar, FFP* tgsw_ntt, CuNTTHandler<> ntt)
{
    static const uint32_t decomp_bits = cuFHE_DEF_Bgbit;
    static const uint32_t decomp_mask = (1 << decomp_bits) - 1;
    static const int32_t decomp_half = 1 << (decomp_bits - 1);
    static const uint32_t decomp_offset =
        (0x1u << 31) + (0x1u << (31 - decomp_bits));
    uint32_t tid = ThisThreadRankInBlock();
    uint32_t bdim = ThisBlockSize();

    // temp[2] = sh_acc[2] * (x^exp - 1)
    // sh_acc_ntt[0, 1] = Decomp(temp[0])
    // sh_acc_ntt[2, 3] = Decomp(temp[1])
    // This algorithm is tested in cpp.
    Torus temp;
#pragma unroll
    for (int i = tid; i < 1024; i += bdim) {
        uint32_t cmp = (uint32_t)(i < (a_bar & 1023));
        uint32_t neg = -(cmp ^ (a_bar >> 10));
        uint32_t pos = -((1 - cmp) ^ (a_bar >> 10));
#pragma unroll
        for (int j = 0; j < 2; j++) {
            temp = tlwe[(j << 10) | ((i - a_bar) & 1023)];
            temp = (temp & pos) + ((-temp) & neg);
            temp -= tlwe[(j << 10) | i];
            // decomp temp
            temp += decomp_offset;
            sh_acc_ntt[(2 * j) * 1024 + i] = FFP(Torus(
                ((temp >> (32 - decomp_bits)) & decomp_mask) - decomp_half));
            sh_acc_ntt[(2 * j + 1) * 1024 + i] =
                FFP(Torus(((temp >> (32 - 2 * decomp_bits)) & decomp_mask) -
                          decomp_half));
        }
    }
    __syncthreads();  // must

    // 4 NTTs with 512 threads.
    // Input/output/buffer use the same shared memory location.
    if (tid < 512) {
        FFP* tar = &sh_acc_ntt[tid >> 7 << 10];
        ntt.NTT<FFP>(tar, tar, tar, tid >> 7 << 7);
    }
    else {  // must meet 4 sync made by NTTInv
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
    }
    __syncthreads();

// Multiply with bootstrapping key in global memory.
#pragma unroll
    for (int i = tid; i < 1024; i += bdim) {
        sh_res_ntt[4096 + i] = 0;
#pragma unroll
        for (int j = 0; j < 4; j++)
            sh_res_ntt[4096 + i] +=
                sh_acc_ntt[j * 1024 + i] * tgsw_ntt[((2 * j + 1) << 10) + i];
    }
    __syncthreads();  // new
#pragma unroll
    for (int i = tid; i < 1024; i += bdim) {
        FFP temp = 0;
#pragma unroll
        for (int j = 0; j < 4; j++)
            temp += sh_acc_ntt[j * 1024 + i] * tgsw_ntt[((2 * j) << 10) + i];
        sh_res_ntt[i] = temp;
    }
    __syncthreads();  // must

    // 2 NTTInvs and add acc with 256 threads.
    if (tid < 256) {
        FFP* src = &sh_res_ntt[tid >> 7 << 12];
        ntt.NTTInvAdd<Torus>(&tlwe[tid >> 7 << 10], src, src, tid >> 7 << 7);
    }
    else {  // must meet 4 sync made by NTTInv
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
    }
    __syncthreads();  // must
}

__global__ void __Bootstrap__(Torus* out, Torus* in, Torus mu, FFP* bk,
                              Torus* ksk, CuNTTHandler<> ntt)
{
    //  Assert(bk.k() == 1);
    //  Assert(bk.l() == 2);
    //  Assert(bk.n() == 1024);
    __shared__ FFP sh[6 * 1024];
    //  FFP* sh_acc_ntt[4] = { sh, sh + 1024, sh + 2048, sh + 3072 };
    //  FFP* sh_res_ntt[2] = { sh, sh + 4096 };
    Torus* tlwe = (Torus*)&sh[5120];

    // test vector
    // acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar = 2048 - ModSwitch2048(in[500]);
    register uint32_t tid = ThisThreadRankInBlock();
    register uint32_t bdim = ThisBlockSize();
    register uint32_t cmp, neg, pos;
#pragma unroll
    for (int i = tid; i < 1024; i += bdim) {
        tlwe[i] = 0;  // part a
        if (bar == 2048)
            tlwe[i + 1024] = mu;
        else {
            cmp = (uint32_t)(i < (bar & 1023));
            neg = -(cmp ^ (bar >> 10));
            pos = -((1 - cmp) ^ (bar >> 10));
            tlwe[i + 1024] = (mu & pos) + ((-mu) & neg);  // part b
        }
    }
    __syncthreads();
// accumulate
#pragma unroll
    for (int i = 0; i < cuFHE_DEF_n; i++) {  // n iterations
        bar = ModSwitch2048(in[i]);
        Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt);
    }

    static const uint32_t lwe_n = 500;
    static const uint32_t tlwe_n = 1024;
    static const uint32_t ks_bits = 2;
    static const uint32_t ks_size = 8;
    KeySwitch<cuFHE_DEF_n, cuFHE_DEF_N, 2, 8>(out, tlwe, ksk);
    __threadfence();
}

__global__ void __BootstrapTLWE2TRLWE__(Torus* out, Torus* in, Torus mu,
                                        FFP* bk, Torus* ksk, CuNTTHandler<> ntt)
{
    //  Assert(bk.k() == 1);
    //  Assert(bk.l() == 2);
    //  Assert(bk.n() == 1024);
    __shared__ FFP sh[6 * 1024];
    //  FFP* sh_acc_ntt[4] = { sh, sh + 1024, sh + 2048, sh + 3072 };
    //  FFP* sh_res_ntt[2] = { sh, sh + 4096 };
    Torus* tlwe = (Torus*)&sh[5120];

    // test vector
    // acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar = 2048 - ModSwitch2048(in[500]);
    register uint32_t tid = ThisThreadRankInBlock();
    register uint32_t bdim = ThisBlockSize();
    register uint32_t cmp, neg, pos;
#pragma unroll
    for (int i = tid; i < 1024; i += bdim) {
        tlwe[i] = 0;  // part a
        if (bar == 2048)
            tlwe[i + 1024] = mu;
        else {
            cmp = (uint32_t)(i < (bar & 1023));
            neg = -(cmp ^ (bar >> 10));
            pos = -((1 - cmp) ^ (bar >> 10));
            tlwe[i + 1024] = (mu & pos) + ((-mu) & neg);  // part b
        }
    }
    __syncthreads();
// accumulate
#pragma unroll
    for (int i = 0; i < cuFHE_DEF_n; i++) {  // n iterations
        bar = ModSwitch2048(in[i]);
        Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt);
    }
    __syncthreads();
    for (int i = 0; i < 2 * cuFHE_DEF_N; i++) {
        out[i] = tlwe[i];
    }
    __threadfence();
}

__global__ void __NandBootstrap__(Torus* out, Torus* in0, Torus* in1, Torus mu,
                                  Torus fix, FFP* bk, Torus* ksk,
                                  CuNTTHandler<> ntt)
{
    __shared__ FFP
        sh[(2 * cuFHE_DEF_l + 2) * cuFHE_DEF_N];  // This is V100's MAX
    // Use Last section to hold tlwe. This may to make these data in serial
    Torus* tlwe = (Torus*)&sh[(2 * cuFHE_DEF_l + 1) * cuFHE_DEF_N];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar =
        2 * cuFHE_DEF_N -
        ModSwitch2048(fix - in0[cuFHE_DEF_n] - in1[cuFHE_DEF_n]);
    RotatedTestVector<cuFHE_DEF_n, cuFHE_DEF_N, cuFHE_DEF_Nbit>(tlwe, bar, mu);

// accumulate
#pragma unroll
    for (int i = 0; i < cuFHE_DEF_n; i++) {  // 500 iterations
        bar = ModSwitch2048(0 - in0[i] - in1[i]);
        Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt);
    }
    KeySwitch<cuFHE_DEF_n, cuFHE_DEF_N, 2, 8>(out, tlwe, ksk);
    __threadfence();
}

__global__ void __OrBootstrap__(Torus* out, Torus* in0, Torus* in1, Torus mu,
                                Torus fix, FFP* bk, Torus* ksk,
                                CuNTTHandler<> ntt)
{
    __shared__ FFP sh[6 * 1024];
    Torus* tlwe = (Torus*)&sh[5120];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar =
        2 * cuFHE_DEF_N -
        ModSwitch2048(fix + in0[cuFHE_DEF_n] + in1[cuFHE_DEF_n]);
    RotatedTestVector<cuFHE_DEF_n, cuFHE_DEF_N, cuFHE_DEF_Nbit>(tlwe, bar, mu);

// accumulate
#pragma unroll
    for (int i = 0; i < 500; i++) {  // 500 iterations
        bar = ModSwitch2048(0 + in0[i] + in1[i]);
        Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt);
    }
    KeySwitch<cuFHE_DEF_n, cuFHE_DEF_N, 2, 8>(out, tlwe, ksk);
    __threadfence();
}

__global__ void __OrYNBootstrap__(Torus* out, Torus* in0, Torus* in1, Torus mu,
                                  Torus fix, FFP* bk, Torus* ksk,
                                  CuNTTHandler<> ntt)
{
    __shared__ FFP sh[6 * 1024];
    Torus* tlwe = (Torus*)&sh[5120];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar =
        2 * cuFHE_DEF_N -
        ModSwitch2048(fix + in0[cuFHE_DEF_n] - in1[cuFHE_DEF_n]);
    RotatedTestVector<cuFHE_DEF_n, cuFHE_DEF_N, cuFHE_DEF_Nbit>(tlwe, bar, mu);

// accumulate
#pragma unroll
    for (int i = 0; i < cuFHE_DEF_n; i++) {  // 500 iterations
        bar = ModSwitch2048(0 + in0[i] - in1[i]);
        Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt);
    }
    KeySwitch<cuFHE_DEF_n, cuFHE_DEF_N, 2, 8>(out, tlwe, ksk);
    __threadfence();
}

__global__ void __OrNYBootstrap__(Torus* out, Torus* in0, Torus* in1, Torus mu,
                                  Torus fix, FFP* bk, Torus* ksk,
                                  CuNTTHandler<> ntt)
{
    __shared__ FFP sh[6 * 1024];
    Torus* tlwe = (Torus*)&sh[5120];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar =
        2 * cuFHE_DEF_N -
        ModSwitch2048(fix - in0[cuFHE_DEF_n] + in1[cuFHE_DEF_n]);
    RotatedTestVector<cuFHE_DEF_n, cuFHE_DEF_N, cuFHE_DEF_Nbit>(tlwe, bar, mu);

// accumulate
#pragma unroll
    for (int i = 0; i < cuFHE_DEF_n; i++) {  // 500 iterations
        bar = ModSwitch2048(0 - in0[i] + in1[i]);
        Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt);
    }
    KeySwitch<cuFHE_DEF_n, cuFHE_DEF_N, 2, 8>(out, tlwe, ksk);
    __threadfence();
}

__global__ void __AndBootstrap__(Torus* out, Torus* in0, Torus* in1, Torus mu,
                                 Torus fix, FFP* bk, Torus* ksk,
                                 CuNTTHandler<> ntt)
{
    __shared__ FFP
        sh[(2 * cuFHE_DEF_l + 2) * cuFHE_DEF_N];  // This is V100's MAX
    Torus* tlwe = (Torus*)&sh[(2 * cuFHE_DEF_l + 1) * cuFHE_DEF_N];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar =
        2 * cuFHE_DEF_N -
        ModSwitch2048(fix + in0[cuFHE_DEF_n] + in1[cuFHE_DEF_n]);
    RotatedTestVector<cuFHE_DEF_n, cuFHE_DEF_N, cuFHE_DEF_Nbit>(tlwe, bar, mu);

// accumulate
#pragma unroll
    for (int i = 0; i < cuFHE_DEF_n; i++) {  // 500 iterations
        bar = ModSwitch2048(0 + in0[i] + in1[i]);
        Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt);
    }
    KeySwitch<cuFHE_DEF_n, cuFHE_DEF_N, 2, 8>(out, tlwe, ksk);
    __threadfence();
}

__global__ void __AndYNBootstrap__(Torus* out, Torus* in0, Torus* in1, Torus mu,
                                   Torus fix, FFP* bk, Torus* ksk,
                                   CuNTTHandler<> ntt)
{
    __shared__ FFP
        sh[(2 * cuFHE_DEF_l + 2) * cuFHE_DEF_N];  // This is V100's MAX
    Torus* tlwe = (Torus*)&sh[(2 * cuFHE_DEF_l + 1) * cuFHE_DEF_N];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar =
        2 * cuFHE_DEF_N -
        ModSwitch2048(fix + in0[cuFHE_DEF_n] - in1[cuFHE_DEF_n]);
    RotatedTestVector<cuFHE_DEF_n, cuFHE_DEF_N, cuFHE_DEF_Nbit>(tlwe, bar, mu);

// accumulate
#pragma unroll
    for (int i = 0; i < cuFHE_DEF_n; i++) {  // 500 iterations
        bar = ModSwitch2048(0 + in0[i] - in1[i]);
        Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt);
    }
    KeySwitch<cuFHE_DEF_n, cuFHE_DEF_N, 2, 8>(out, tlwe, ksk);
    __threadfence();
}

__global__ void __AndNYBootstrap__(Torus* out, Torus* in0, Torus* in1, Torus mu,
                                   Torus fix, FFP* bk, Torus* ksk,
                                   CuNTTHandler<> ntt)
{
    __shared__ FFP
        sh[(2 * cuFHE_DEF_l + 2) * cuFHE_DEF_N];  // This is V100's MAX
    Torus* tlwe = (Torus*)&sh[(2 * cuFHE_DEF_l + 1) * cuFHE_DEF_N];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar =
        2 * cuFHE_DEF_N -
        ModSwitch2048(fix - in0[cuFHE_DEF_n] + in1[cuFHE_DEF_n]);
    RotatedTestVector<cuFHE_DEF_n, cuFHE_DEF_N, cuFHE_DEF_Nbit>(tlwe, bar, mu);

// accumulate
#pragma unroll
    for (int i = 0; i < cuFHE_DEF_n; i++) {  // 500 iterations
        bar = ModSwitch2048(0 - in0[i] + in1[i]);
        Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt);
    }
    KeySwitch<cuFHE_DEF_n, cuFHE_DEF_N, 2, 8>(out, tlwe, ksk);
    __threadfence();
}

__global__ void __NorBootstrap__(Torus* out, Torus* in0, Torus* in1, Torus mu,
                                 Torus fix, FFP* bk, Torus* ksk,
                                 CuNTTHandler<> ntt)
{
    __shared__ FFP sh[6 * 1024];
    Torus* tlwe = (Torus*)&sh[5120];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar = 2048 - ModSwitch2048(fix - in0[500] - in1[500]);
    RotatedTestVector<cuFHE_DEF_n, cuFHE_DEF_N>(tlwe, bar, mu);

// accumulate
#pragma unroll
    for (int i = 0; i < 500; i++) {  // 500 iterations
        bar = ModSwitch2048(0 - in0[i] - in1[i]);
        Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt);
    }
    KeySwitch<cuFHE_DEF_n, cuFHE_DEF_N, 2, 8>(out, tlwe, ksk);
    __threadfence();
}

__global__ void __XorBootstrap__(Torus* out, Torus* in0, Torus* in1, Torus mu,
                                 Torus fix, FFP* bk, Torus* ksk,
                                 CuNTTHandler<> ntt)
{
    __shared__ FFP sh[6 * 1024];
    Torus* tlwe = (Torus*)&sh[5120];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar =
        2048 - ModSwitch2048(fix + 2 * in0[500] + 2 * in1[500]);
    RotatedTestVector<cuFHE_DEF_n, cuFHE_DEF_N>(tlwe, bar, mu);

// accumulate
#pragma unroll
    for (int i = 0; i < 500; i++) {  // 500 iterations
        bar = ModSwitch2048(0 + 2 * in0[i] + 2 * in1[i]);
        Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt);
    }
    KeySwitch<cuFHE_DEF_n, cuFHE_DEF_N, 2, 8>(out, tlwe, ksk);
    __threadfence();
}

__global__ void __XnorBootstrap__(Torus* out, Torus* in0, Torus* in1, Torus mu,
                                  Torus fix, FFP* bk, Torus* ksk,
                                  CuNTTHandler<> ntt)
{
    __shared__ FFP sh[6 * 1024];
    Torus* tlwe = (Torus*)&sh[5120];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar =
        2048 - ModSwitch2048(fix - 2 * in0[500] - 2 * in1[500]);
    RotatedTestVector<cuFHE_DEF_n, cuFHE_DEF_N>(tlwe, bar, mu);

// accumulate
#pragma unroll
    for (int i = 0; i < cuFHE_DEF_n; i++) {  // 500 iterations
        bar = ModSwitch2048(0 - 2 * in0[i] - 2 * in1[i]);
        Accumulate(tlwe, sh, sh, bar, bk + (i << 13), ntt);
    }
    KeySwitch<500, 1024, 2, 8>(out, tlwe, ksk);
    __threadfence();
}

__global__ void __CopyBootstrap__(Torus* out, Torus* in)
{
    uint32_t tid = ThisThreadRankInBlock();
    out[tid] = in[tid];
    __syncthreads();
    __threadfence();
}

__global__ void __NotBootstrap__(Torus* out, Torus* in, int n)
{
#pragma unroll
    for (int i = 0; i <= n; i++) {
        out[i] = -in[i];
    }
    __syncthreads();
    __threadfence();
}

// Mux(inc,in1,in0) = inc?in1:in0 = inc&in1 + (!inc)&in0
__global__ void __MuxBootstrap__(Torus* out, Torus* inc, Torus* in1, Torus* in0,
                                 Torus mu, Torus fix, Torus muxfix, FFP* bk,
                                 Torus* ksk, CuNTTHandler<> ntt)
{
    // To use over 48 KiB shared Memory, the dynamic allocation is required.
    extern __shared__ FFP sh[];
    // Use Last section to hold tlwe. This may to make these data in serial
    Torus* tlwe1 = (Torus*)&sh[(2 * cuFHE_DEF_l + 1) * cuFHE_DEF_N];
    Torus* tlwe0 = (Torus*)&sh[(2 * cuFHE_DEF_l + 2) * cuFHE_DEF_N];
    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar =
        2 * cuFHE_DEF_N -
        ModSwitch2048(fix + inc[cuFHE_DEF_n] + in1[cuFHE_DEF_n]);
    RotatedTestVector<cuFHE_DEF_n, cuFHE_DEF_N>(tlwe1, bar, mu);

// accumulate
#pragma unroll
    for (int i = 0; i < cuFHE_DEF_n; i++) {  // 500 iterations
        bar = ModSwitch2048(0 + inc[i] + in1[i]);
        Accumulate(tlwe1, sh, sh, bar, bk + (i << 13), ntt);
    }

    bar = 2 * cuFHE_DEF_N -
          ModSwitch2048(fix - inc[cuFHE_DEF_n] + in0[cuFHE_DEF_n]);

    RotatedTestVector<cuFHE_DEF_n, cuFHE_DEF_N>(tlwe0, bar, mu);

#pragma unroll
    for (int i = 0; i < cuFHE_DEF_n; i++) {  // 500 iterations
        bar = ModSwitch2048(0 - inc[i] + in0[i]);
        Accumulate(tlwe0, sh, sh, bar, bk + (i << 13), ntt);
    }

    volatile uint32_t tid = ThisThreadRankInBlock();
    volatile uint32_t bdim = ThisBlockSize();
#pragma unroll
    for (int i = tid; i <= cuFHE_DEF_N; i += bdim) {
        tlwe1[i] += tlwe0[i];
        if (i == cuFHE_DEF_N) {
            tlwe1[cuFHE_DEF_N] += muxfix;
        }
    }

    __syncthreads();

    KeySwitch<500, 1024, 2, 8>(out, tlwe1, ksk);
    __threadfence();
}

__global__ void __NoiselessTrivial__(Torus* out, Torus pmu)
{
    register uint32_t tid = ThisThreadRankInBlock();
    register uint32_t bdim = ThisBlockSize();
#pragma unroll
    for (int i = tid; i <= cuFHE_DEF_n; i += bdim) {
        if (i == cuFHE_DEF_n)
            out[cuFHE_DEF_n] = pmu;
        else
            out[i] = 0;
    }
    __threadfence();
}

void Bootstrap(LWESample* out, LWESample* in, Torus mu, cudaStream_t st,
               int gpuNum)
{
    dim3 grid(1);
    dim3 block(512);
    __Bootstrap__<<<grid, block, 0, st>>>(
        out->data(), in->data(), mu, bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void BootstrapTLWE2TRLWE(Torus* out, LWESample* in, Torus mu, cudaStream_t st,
                         int gpuNum)
{
    dim3 grid(1);
    dim3 block(512);
    __BootstrapTLWE2TRLWE__<<<grid, block, 0, st>>>(
        out, in->data(), mu, bk_ntts[gpuNum]->data(), ksk_devs[gpuNum]->data(),
        *ntt_handlers[gpuNum]);
    CuCheckError();
}

void NandBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                   Torus fix, cudaStream_t st, int gpuNum)
{
    __NandBootstrap__<<<1, cuFHE_DEF_N / 2, 0, st>>>(
        out->data(), in0->data(), in1->data(), mu, fix, bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void OrBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                 Torus fix, cudaStream_t st, int gpuNum)
{
    __OrBootstrap__<<<1, cuFHE_DEF_N / 2, 0, st>>>(
        out->data(), in0->data(), in1->data(), mu, fix, bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void OrYNBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                   Torus fix, cudaStream_t st, int gpuNum)
{
    __OrYNBootstrap__<<<1, cuFHE_DEF_N / 2, 0, st>>>(
        out->data(), in0->data(), in1->data(), mu, fix, bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void OrNYBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                   Torus fix, cudaStream_t st, int gpuNum)
{
    __OrNYBootstrap__<<<1, cuFHE_DEF_N / 2, 0, st>>>(
        out->data(), in0->data(), in1->data(), mu, fix, bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void AndBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                  Torus fix, cudaStream_t st, int gpuNum)
{
    __AndBootstrap__<<<1, cuFHE_DEF_N / 2, 0, st>>>(
        out->data(), in0->data(), in1->data(), mu, fix, bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void AndYNBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                    Torus fix, cudaStream_t st, int gpuNum)
{
    __AndYNBootstrap__<<<1, cuFHE_DEF_N / 2, 0, st>>>(
        out->data(), in0->data(), in1->data(), mu, fix, bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void AndNYBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                    Torus fix, cudaStream_t st, int gpuNum)
{
    __AndNYBootstrap__<<<1, cuFHE_DEF_N / 2, 0, st>>>(
        out->data(), in0->data(), in1->data(), mu, fix, bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void NorBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                  Torus fix, cudaStream_t st, int gpuNum)
{
    __NorBootstrap__<<<1, cuFHE_DEF_N / 2, 0, st>>>(
        out->data(), in0->data(), in1->data(), mu, fix, bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void XorBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                  Torus fix, cudaStream_t st, int gpuNum)
{
    __XorBootstrap__<<<1, cuFHE_DEF_N / 2, 0, st>>>(
        out->data(), in0->data(), in1->data(), mu, fix, bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void XnorBootstrap(LWESample* out, LWESample* in0, LWESample* in1, Torus mu,
                   Torus fix, cudaStream_t st, int gpuNum)
{
    __XnorBootstrap__<<<1, cuFHE_DEF_N / 2, 0, st>>>(
        out->data(), in0->data(), in1->data(), mu, fix, bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void CopyBootstrap(LWESample* out, LWESample* in, cudaStream_t st, int gpuNum)
{
    __CopyBootstrap__<<<1, cuFHE_DEF_n + 1, 0, st>>>(out->data(), in->data());
    CuCheckError();
}

void NotBootstrap(LWESample* out, LWESample* in, int n, cudaStream_t st,
                  int gpuNum)
{
    __NotBootstrap__<<<1, cuFHE_DEF_N / 2, 0, st>>>(out->data(), in->data(), n);
    CuCheckError();
}

void MuxBootstrap(LWESample* out, LWESample* inc, LWESample* in1,
                  LWESample* in0, Torus mu, Torus fix, Torus muxfix,
                  cudaStream_t st, int gpuNum)
{
    int maxbytes = 98304;  // 96 KB
    cudaFuncSetAttribute(__MuxBootstrap__,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (2 * cuFHE_DEF_l + 3) * cuFHE_DEF_N * sizeof(FFP));
    __MuxBootstrap__<<<1, cuFHE_DEF_N / 2,
                       (2 * cuFHE_DEF_l + 3) * cuFHE_DEF_N * sizeof(FFP), st>>>(
        out->data(), inc->data(), in1->data(), in0->data(), mu, fix, muxfix,
        bk_ntts[gpuNum]->data(), ksk_devs[gpuNum]->data(),
        *ntt_handlers[gpuNum]);
    CuCheckError();
}

void NoiselessTrivial(LWESample* out, int p, Torus mu, cudaStream_t st)
{
    __NoiselessTrivial__<<<1, cuFHE_DEF_n + 1, 0, st>>>(out->data(),
                                                        p ? mu : -mu);
}
}  // namespace cufhe
