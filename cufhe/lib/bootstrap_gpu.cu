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

#include <bits/stdint-uintn.h>
#include <include/cufhe.h>
#include <stdio.h>
#include <unistd.h>
#include <include/bootstrap_gpu.cuh>
#include <include/details/error_gpu.cuh>
#include <include/ntt_gpu/ntt.cuh>

#include <iostream>
#include <limits>
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
    __shared__ FFP sh_temp[lvl1param::n];

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

template <class P>
__device__ inline void KeySwitch(Torus* lwe, Torus* tlwe, Torus* ksk)
{
    constexpr Torus decomp_mask = (1u << P::basebit) - 1;
    constexpr Torus decomp_offset = 1u << (std::numeric_limits<typename P::domainP::T>::digits - 1 - P::t * P::basebit);
    uint32_t tid = ThisThreadRankInBlock();
    uint32_t bdim = ThisBlockSize();
#pragma unroll 0
    for (int i = tid; i <= P::targetP::n; i += bdim) {
	Torus tmp;
    	Torus res = 0;
    	Torus val = 0;
        if (i == P::targetP::n) res = tlwe[P::domainP::n];
#pragma unroll 0
        for (int j = 0; j < P::domainP::n; j++) {
            if (j == 0)
                tmp = tlwe[0];
            else
                tmp = -tlwe[P::domainP::n - j];
            tmp += decomp_offset;
            for (int k = 0; k < P::t; k++) {
                val = (tmp >> (std::numeric_limits<typename P::domainP::T>::digits - (k + 1) * P::basebit)) & decomp_mask;
                if (val != 0)
                    res -= ksk[(j << 14) | (k << 11) | (val << 9) | i];
            }
        }
        lwe[i] = res;
    }
}

template <class P>
__device__ inline void RotatedTestVector(Torus* tlwe, int32_t bar, const typename P::T μ)
{
    // volatile is needed to make register usage of Mux to 128.
    // Reference
    // https://devtalk.nvidia.com/default/topic/466758/cuda-programming-and-performance/tricks-to-fight-register-pressure-or-how-i-got-down-from-29-to-15-registers-/
    volatile uint32_t tid = ThisThreadRankInBlock();
    volatile uint32_t bdim = ThisBlockSize();
    uint32_t cmp, neg, pos;
#pragma unroll
    for (int i = tid; i < P::n; i += bdim) {
        tlwe[i] = 0;  // part a
        if (bar == 2 * P::n)
            tlwe[i + P::n] = μ;
        else {
            cmp = (uint32_t)(i < (bar & (P::n - 1)));
            neg = -(cmp ^ (bar >> P::nbit));
            pos = -((1 - cmp) ^ (bar >> P::nbit));
            tlwe[i + P::n] = (μ & pos) + ((-μ) & neg);  // part b
        }
    }
    __syncthreads();
}

template <class P>
__device__ constexpr typename P::T offsetgen()
{
    typename P::T offset = 0;
    for (int i = 1; i <= P::l; i++)
        offset +=
            P::Bg / 2 *
            (1ULL << (numeric_limits<typename P::T>::digits - i * P::Bgbit));
    return offset;
}

__device__ inline void PolynomialMulByXaiMinusOneAndDecomposition(FFP* decpoly, const Torus* poly, const uint32_t a_bar,const int digit){
    // temp[2] = sh_acc[2] * (x^exp - 1)
    // sh_acc_ntt[0, 1] = Decomp(temp[0])
    // sh_acc_ntt[2, 3] = Decomp(temp[1])
    // This algorithm is tested in cpp.
    uint32_t tid = ThisThreadRankInBlock();
    uint32_t bdim = ThisBlockSize();
    constexpr uint32_t decomp_mask = (1 << lvl1param::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (lvl1param::Bgbit - 1);
    constexpr uint32_t decomp_offset = offsetgen<lvl1param>();
    Torus temp;
#pragma unroll
    for (int i = tid; i < lvl1param::n; i += bdim) {
        uint32_t cmp = (uint32_t)(i < (a_bar & (lvl1param::n - 1)));
        uint32_t neg = -(cmp ^ (a_bar >> lvl1param::nbit));
        uint32_t pos = -((1 - cmp) ^ (a_bar >> lvl1param::nbit));
            temp = poly[(i - a_bar) & (lvl1param::n - 1)];
            temp = (temp & pos) + ((-temp) & neg);
            temp -= poly[i];
            // decomp temp
            temp += decomp_offset;
            decpoly[i] = FFP(Torus(
                ((temp >> (32 - (digit+1) * lvl1param::Bgbit)) & decomp_mask) - decomp_half));
    }
    __syncthreads();  // must
}

__device__ inline void Accumulate(Torus* tlwe, FFP* sh_res_ntt, FFP* decpoly,
                           const uint32_t a_bar, const FFP* tgsw_ntt, const CuNTTHandler<> ntt)
{
    uint32_t tid = ThisThreadRankInBlock();
    uint32_t bdim = ThisBlockSize();

    PolynomialMulByXaiMinusOneAndDecomposition(decpoly, &tlwe[0], a_bar, 0);

    // 1 NTTs with 128 threads.
    // Input/output/buffer use the same shared memory location.
    if (tid < 128) {
        FFP* tar = &decpoly[0];
        ntt.NTT<FFP>(tar, tar, tar, 0);
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
    for (int i = tid; i < lvl1param::n; i += bdim) {
        sh_res_ntt[i] =
            decpoly[i] * tgsw_ntt[((2 * 0 + 0) << lvl1param::nbit) + i];
        sh_res_ntt[i+lvl1param::n] = decpoly[i] * tgsw_ntt[((2 * 0 + 1) << lvl1param::nbit) + i];
    }
    __syncthreads();
#pragma unroll
    for (int digit = 1;digit<lvl1param::l;digit++){
        PolynomialMulByXaiMinusOneAndDecomposition(decpoly, &tlwe[0], a_bar, digit);

            // 1 NTTs with 128 threads.
            // Input/output/buffer use the same shared memory location.
            if (tid < 128) {
                FFP* tar = &decpoly[0];
                ntt.NTT<FFP>(tar, tar, tar, 0);
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
            for (int i = tid; i < lvl1param::n; i += bdim) {
                sh_res_ntt[i] +=
                    decpoly[i] * tgsw_ntt[((2 * digit + 0) << lvl1param::nbit) + i];
                sh_res_ntt[i+lvl1param::n] += decpoly[i] * tgsw_ntt[((2 * digit + 1) << lvl1param::nbit) + i];
            }
    }

    #pragma unroll
    for (int digit = 0;digit<lvl1param::l;digit++){
        PolynomialMulByXaiMinusOneAndDecomposition(decpoly, &tlwe[lvl1param::n], a_bar, digit);

            // 1 NTTs with 128 threads.
            // Input/output/buffer use the same shared memory location.
            if (tid < 128) {
                FFP* tar = &decpoly[0];
                ntt.NTT<FFP>(tar, tar, tar, 0);
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
            for (int i = tid; i < lvl1param::n; i += bdim) {
                sh_res_ntt[i] +=
                    decpoly[i] * tgsw_ntt[((2 * (digit+lvl1param::l) + 0) << lvl1param::nbit) + i];
                sh_res_ntt[i+lvl1param::n] += decpoly[i] * tgsw_ntt[((2 * (digit+lvl1param::l) + 1) << lvl1param::nbit) + i];
            }
    }

    // 1 NTTInvs and add acc with 128 threads.
    if (tid < 128) {
        FFP* src = &sh_res_ntt[0];
        ntt.NTTInvAdd<Torus>(&tlwe[0], src, src, 0);
    }
    else {  // must meet 4 sync made by NTTInv
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
    }
    __syncthreads();  // must
    // 1 NTTInvs and add acc with 128 threads.
    if (tid < 128) {
        FFP* src = &sh_res_ntt[lvl1param::n];
        ntt.NTTInvAdd<Torus>(&tlwe[lvl1param::n], src, src, 0);
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
    //  Assert(bk.n() == lvl1param::n);
    __shared__ FFP sh[(2 * lvl1param::l + 2) * lvl1param::n];
    //  FFP* sh_acc_ntt[4] = { sh, sh + 1024, sh + 2048, sh + 3072 };
    //  FFP* sh_res_ntt[2] = { sh, sh + 4096 };
    Torus* tlwe = (Torus*)&sh[(2 * lvl1param::l + 1) * lvl1param::n];

    // test vector
    // acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar = 2 * lvl1param::n - ModSwitch2048(in[lvl1param::n]);
    register uint32_t tid = ThisThreadRankInBlock();
    register uint32_t bdim = ThisBlockSize();
    register uint32_t cmp, neg, pos;
#pragma unroll
    for (int i = tid; i < lvl1param::n; i += bdim) {
        tlwe[i] = 0;  // part a
        if (bar == 2 * lvl1param::n)
            tlwe[i + lvl1param::n] = mu;
        else {
            cmp = (uint32_t)(i < (bar & (lvl1param::n - 1)));
            neg = -(cmp ^ (bar >> lvl1param::nbit));
            pos = -((1 - cmp) ^ (bar >> lvl1param::nbit));
            tlwe[i + lvl1param::n] = (mu & pos) + ((-mu) & neg);  // part b
        }
    }
    __syncthreads();
// accumulate
#pragma unroll
    for (int i = 0; i < lvl1param::n; i++) {  // n iterations
        bar = ModSwitch2048(in[i]);
        Accumulate(tlwe, sh, sh, bar, bk + (i << lvl1param::nbit)*2*2*lvl1param::l, ntt);
    }

    KeySwitch<lvl10param>(out, tlwe, ksk);
    __threadfence();
}

__global__ void __SEandKS__(Torus* out, Torus* in, FFP* bk, Torus* ksk,
                            CuNTTHandler<> ntt)
{
    KeySwitch<lvl10param>(out, in, ksk);
    __threadfence();
}

__global__ void __BootstrapTLWE2TRLWE__(Torus* out, Torus* in, Torus mu,
                                        FFP* bk, Torus* ksk, CuNTTHandler<> ntt)
{
    //  Assert(bk.k() == 1);
    //  Assert(bk.l() == 2);
    //  Assert(bk.n() == lvl1param::n);
    __shared__ FFP sh[(2 * lvl1param::l + 2) * lvl1param::n];
    //  FFP* sh_acc_ntt[4] = { sh, sh + 1024, sh + 2048, sh + 3072 };
    //  FFP* sh_res_ntt[2] = { sh, sh + 4096 };
    Torus* tlwe = (Torus*)&sh[(2 * lvl1param::l + 1) * lvl1param::n];

    // test vector
    // acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar = 2048 - ModSwitch2048(in[lvl1param::n]);
    register uint32_t tid = ThisThreadRankInBlock();
    register uint32_t bdim = ThisBlockSize();
    register uint32_t cmp, neg, pos;
#pragma unroll
    for (int i = tid; i < lvl1param::n; i += bdim) {
        tlwe[i] = 0;  // part a
        if (bar == 2048)
            tlwe[i + lvl1param::n] = mu;
        else {
            cmp = (uint32_t)(i < (bar & (lvl1param::n - 1)));
            neg = -(cmp ^ (bar >> lvl1param::nbit));
            pos = -((1 - cmp) ^ (bar >> lvl1param::nbit));
            tlwe[i + lvl1param::n] = (mu & pos) + ((-mu) & neg);  // part b
        }
    }
    __syncthreads();
// accumulate
#pragma unroll
    for (int i = 0; i < lvl1param::n; i++) {  // n iterations
        bar = ModSwitch2048(in[i]);
        Accumulate(tlwe, sh, sh, bar, bk + (i << lvl1param::nbit)*2*2*lvl1param::l, ntt);
    }
    __syncthreads();
    for (int i = 0; i < 2 * lvl1param::n; i++) {
        out[i] = tlwe[i];
    }
    __threadfence();
}

template<int casign, int cbsign, typename lvl0param::T offset>
__device__ void __HomGate__(Torus* out, Torus* in0, Torus* in1, 
                                  FFP* bk, Torus* ksk,
                                  CuNTTHandler<> ntt)
{
    __shared__ FFP
        sh[(2 + 2) * lvl1param::n];  // This is V100's MAX
    FFP* sh_acc_ntt = &sh[0];
    FFP* decpoly = &sh[2*lvl1param::n];
    // Use Last section to hold tlwe. This may to make these data in serial
    Torus* tlwe = (Torus*)&sh[(2 + 1) * lvl1param::n];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar =
        2 * lvl1param::n -
        ModSwitch2048(offset + casign * in0[lvl1param::n] + casign * in1[lvl1param::n]);
    RotatedTestVector<lvl1param>(tlwe, bar, lvl1param::μ);

// accumulate
#pragma unroll
    for (int i = 0; i < lvl1param::n; i++) {  // lvl1param::n iterations
        bar = ModSwitch2048(0 - in0[i] - in1[i]);
        Accumulate(tlwe, sh_acc_ntt, decpoly, bar, bk + (i << lvl1param::nbit)*2*2*lvl1param::l, ntt);
    }
    KeySwitch<lvl10param>(out, tlwe, ksk);
    __threadfence();
}

__global__ void __NandBootstrap__(Torus* out, Torus* in0, Torus* in1, 
                                  FFP* bk, Torus* ksk,
                                  CuNTTHandler<> ntt)
{
    __HomGate__<-1, -1, lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

__global__ void __NorBootstrap__(Torus* out, Torus* in0, Torus* in1, 
                                 FFP* bk, Torus* ksk,
                                 CuNTTHandler<> ntt)
{
    __HomGate__<-1, -1, -lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

__global__ void __XnorBootstrap__(Torus* out, Torus* in0, Torus* in1, FFP* bk, Torus* ksk,
                                  CuNTTHandler<> ntt)
{
    __HomGate__<-2, -2, -2*lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

__global__ void __AndBootstrap__(Torus* out, Torus* in0, Torus* in1,  FFP* bk, Torus* ksk,
                                 CuNTTHandler<> ntt)
{
    __HomGate__<1, 1, -lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

__global__ void __OrBootstrap__(Torus* out, Torus* in0, Torus* in1, FFP* bk, Torus* ksk,
                                CuNTTHandler<> ntt)
{
    __HomGate__<1, 1, lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

__global__ void __XorBootstrap__(Torus* out, Torus* in0, Torus* in1, FFP* bk, Torus* ksk,
                                 CuNTTHandler<> ntt)
{
    __HomGate__<2, 2, lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

__global__ void __AndNYBootstrap__(Torus* out, Torus* in0, Torus* in1, FFP* bk, Torus* ksk,
                                   CuNTTHandler<> ntt)
{
    __HomGate__<-1, 1, -lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

__global__ void __AndYNBootstrap__(Torus* out, Torus* in0, Torus* in1, FFP* bk, Torus* ksk,
                                   CuNTTHandler<> ntt)
{
    __HomGate__<1, -1, -lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

__global__ void __OrNYBootstrap__(Torus* out, Torus* in0, Torus* in1, FFP* bk, Torus* ksk,
                                  CuNTTHandler<> ntt)
{
    __HomGate__<-1, 1, lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

__global__ void __OrYNBootstrap__(Torus* out, Torus* in0, Torus* in1, FFP* bk, Torus* ksk,
                                  CuNTTHandler<> ntt)
{
    __HomGate__<1, -1, lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
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
    // Use Last section to hold tlwe. This may make these data in serial.
    Torus* tlwe1 = (Torus*)&sh[(2 * lvl1param::l + 1) * lvl1param::n];
    Torus* tlwe0 = (Torus*)&sh[(2 * lvl1param::l + 2) * lvl1param::n];
    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register int32_t bar =
        2 * lvl1param::n -
        ModSwitch2048(fix + inc[lvl1param::n] + in1[lvl1param::n]);
    RotatedTestVector<lvl1param>(tlwe1, bar, mu);

// accumulate
#pragma unroll
    for (int i = 0; i < lvl1param::n; i++) {  // lvl1param::n iterations
        bar = ModSwitch2048(0 + inc[i] + in1[i]);
        Accumulate(tlwe1, sh, sh, bar, bk + (i << lvl1param::nbit)*2*2*lvl1param::l, ntt);
    }

    bar = 2 * lvl1param::n -
          ModSwitch2048(fix - inc[lvl1param::n] + in0[lvl1param::n]);

    RotatedTestVector<lvl1param>(tlwe0, bar, mu);

#pragma unroll
    for (int i = 0; i < lvl1param::n; i++) {  // lvl1param::n iterations
        bar = ModSwitch2048(0 - inc[i] + in0[i]);
        Accumulate(tlwe0, sh, sh, bar, bk + (i << lvl1param::nbit)*2*2*lvl1param::l, ntt);
    }

    volatile uint32_t tid = ThisThreadRankInBlock();
    volatile uint32_t bdim = ThisBlockSize();
#pragma unroll
    for (int i = tid; i <= lvl1param::n; i += bdim) {
        tlwe1[i] += tlwe0[i];
        if (i == lvl1param::n) {
            tlwe1[lvl1param::n] += muxfix;
        }
    }

    __syncthreads();

    KeySwitch<lvl10param>(out, tlwe1, ksk);
    __threadfence();
}

__global__ void __NoiselessTrivial__(Torus* out, Torus pmu)
{
    register uint32_t tid = ThisThreadRankInBlock();
    register uint32_t bdim = ThisBlockSize();
#pragma unroll
    for (int i = tid; i <= lvl1param::n; i += bdim) {
        if (i == lvl1param::n)
            out[lvl1param::n] = pmu;
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

void SEandKS(LWESample* out, Torus* in, cudaStream_t st, int gpuNum)
{
    dim3 grid(1);
    dim3 block(512);
    __SEandKS__<<<grid, block, 0, st>>>(
        out->data(), in, bk_ntts[gpuNum]->data(), ksk_devs[gpuNum]->data(),
        *ntt_handlers[gpuNum]);
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

void NandBootstrap(LWESample* out, LWESample* in0, LWESample* in1, cudaStream_t st, int gpuNum)
{
    __NandBootstrap__<<<1,lvl1param::n>>NTT_THRED_UNITBIT, 0, st>>>(
        out->data(), in0->data(), in1->data(), bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void OrBootstrap(LWESample* out, LWESample* in0, LWESample* in1, cudaStream_t st, int gpuNum)
{
    __OrBootstrap__<<<1,lvl1param::n>>NTT_THRED_UNITBIT, 0, st>>>(
        out->data(), in0->data(), in1->data(), bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void OrYNBootstrap(LWESample* out, LWESample* in0, LWESample* in1, cudaStream_t st, int gpuNum)
{
    __OrYNBootstrap__<<<1,lvl1param::n>>NTT_THRED_UNITBIT, 0, st>>>(
        out->data(), in0->data(), in1->data(), bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void OrNYBootstrap(LWESample* out, LWESample* in0, LWESample* in1, cudaStream_t st, int gpuNum)
{
    __OrNYBootstrap__<<<1,lvl1param::n>>NTT_THRED_UNITBIT, 0, st>>>(
        out->data(), in0->data(), in1->data(), bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void AndBootstrap(LWESample* out, LWESample* in0, LWESample* in1, cudaStream_t st, int gpuNum)
{
    __AndBootstrap__<<<1,lvl1param::n>>NTT_THRED_UNITBIT, 0, st>>>(
        out->data(), in0->data(), in1->data(), bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void AndYNBootstrap(LWESample* out, LWESample* in0, LWESample* in1, cudaStream_t st, int gpuNum)
{
    __AndYNBootstrap__<<<1,lvl1param::n>>NTT_THRED_UNITBIT, 0, st>>>(
        out->data(), in0->data(), in1->data(), bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void AndNYBootstrap(LWESample* out, LWESample* in0, LWESample* in1, cudaStream_t st, int gpuNum)
{
    __AndNYBootstrap__<<<1,lvl1param::n>>NTT_THRED_UNITBIT, 0, st>>>(
        out->data(), in0->data(), in1->data(), bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void NorBootstrap(LWESample* out, LWESample* in0, LWESample* in1,  cudaStream_t st, int gpuNum)
{
    __NorBootstrap__<<<1,lvl1param::n>>NTT_THRED_UNITBIT, 0, st>>>(
        out->data(), in0->data(), in1->data(), bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void XorBootstrap(LWESample* out, LWESample* in0, LWESample* in1,  cudaStream_t st, int gpuNum)
{
    __XorBootstrap__<<<1,lvl1param::n>>NTT_THRED_UNITBIT, 0, st>>>(
        out->data(), in0->data(), in1->data(), bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void XnorBootstrap(LWESample* out, LWESample* in0, LWESample* in1, cudaStream_t st, int gpuNum)
{
    __XnorBootstrap__<<<1,lvl1param::n>>NTT_THRED_UNITBIT, 0, st>>>(
        out->data(), in0->data(), in1->data(), bk_ntts[gpuNum]->data(),
        ksk_devs[gpuNum]->data(), *ntt_handlers[gpuNum]);
    CuCheckError();
}

void CopyBootstrap(LWESample* out, LWESample* in, cudaStream_t st, int gpuNum)
{
    __CopyBootstrap__<<<1, lvl0param::n + 1, 0, st>>>(out->data(), in->data());
    CuCheckError();
}

void NotBootstrap(LWESample* out, LWESample* in, int n, cudaStream_t st,
                  int gpuNum)
{
    __NotBootstrap__<<<1,lvl1param::n>>NTT_THRED_UNITBIT, 0, st>>>(out->data(), in->data(), n);
    CuCheckError();
}

void MuxBootstrap(LWESample* out, LWESample* inc, LWESample* in1,
                  LWESample* in0, Torus mu, Torus fix, Torus muxfix,
                  cudaStream_t st, int gpuNum)
{
    const int maxbytes = 98304;  // 96 KB
    cudaFuncSetAttribute(__MuxBootstrap__,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (2 * lvl1param::l + 3) * lvl1param::n * sizeof(FFP));
    __MuxBootstrap__<<<1,lvl1param::n>>NTT_THRED_UNITBIT,
                       (2 * lvl1param::l + 3) * lvl1param::n * sizeof(FFP), st>>>(
        out->data(), inc->data(), in1->data(), in0->data(), mu, fix, muxfix,
        bk_ntts[gpuNum]->data(), ksk_devs[gpuNum]->data(),
        *ntt_handlers[gpuNum]);
    CuCheckError();
}

void NoiselessTrivial(LWESample* out, int p, Torus mu, cudaStream_t st)
{
    __NoiselessTrivial__<<<1, lvl1param::n + 1, 0, st>>>(out->data(),
                                                        p ? mu : -mu);
}
}  // namespace cufhe
