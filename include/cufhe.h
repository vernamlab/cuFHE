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

/**
 * @file cufhe.h
 * @brief This is the user API of the cuFHE library.
 *        It hides most of the contents in the developer API and
 *        only provides essential data structures and functions.
 */

#pragma once

#include <bits/stdint-uintn.h>
#include <math.h>
#include <time.h>

#include <array>
#include <iostream>
#include <vector>

#include "../thirdparties/TFHEpp/include/tfhe++.hpp"
#include "details/allocator.h"
#include "ntt_gpu/ntt_ffp.cuh"

namespace cufhe {
using namespace TFHEpp;

/*****************************
 * Parameters *
 *****************************/

// Implementation dependent parameter
constexpr uint32_t NTT_THRED_UNITBIT =
    3;  // How many threads works as one group in NTT algorithm.

/*****************************
 * Essential Data Structures *
 *****************************/

struct Param {
    uint32_t lwe_n_;
    uint32_t tlwe_n_;
    uint32_t tlwe_k_;
    uint32_t tgsw_decomp_bits_;
    uint32_t tgsw_decomp_size_;
    uint32_t keyswitching_decomp_bits_;
    uint32_t keyswitching_decomp_size_;
    double lwe_noise_;
    double tlwe_noise_;

    Param()
        : lwe_n_(lvl0param::n),
          tlwe_n_(lvl1param::n),
          tlwe_k_(1),
          tgsw_decomp_bits_(lvl1param::Bg),
          tgsw_decomp_size_(lvl1param::l),
          keyswitching_decomp_bits_(lvl10param::basebit),
          keyswitching_decomp_size_(lvl10param::t),
          lwe_noise_(lvl0param::α),
          tlwe_noise_(lvl1param::α){};

    Param(uint32_t lwe_n, uint32_t tlwe_n, uint32_t tlwe_k,
          uint32_t tgsw_decomp_bits, uint32_t tgsw_decomp_size,
          uint32_t keyswitching_decomp_bits, uint32_t keyswitching_decomp_size,
          double lwe_noise, double tlwe_noise)
        : lwe_n_(lwe_n),
          tlwe_n_(tlwe_n),
          tlwe_k_(tlwe_k),
          tgsw_decomp_bits_(tgsw_decomp_bits),
          tgsw_decomp_size_(tgsw_decomp_size),
          keyswitching_decomp_bits_(keyswitching_decomp_bits),
          keyswitching_decomp_size_(keyswitching_decomp_size),
          lwe_noise_(lwe_noise),
          tlwe_noise_(tlwe_noise){};
};

Param* GetDefaultParam();

/** Ciphertext. */
struct Ctxt {
    Ctxt();
    ~Ctxt();
    Ctxt(const Ctxt& that) = delete;
    Ctxt& operator=(const Ctxt& that) = delete;

    TFHEpp::TLWE<TFHEpp::lvl0param> tlwehost;

    std::vector<TFHEpp::lvl0param::T*> tlwedevices;
};

/** TRLWE holder */
struct cuFHETRLWElvl1 {
    TFHEpp::TRLWE<TFHEpp::lvl1param> trlwehost;
    std::vector<TFHEpp::lvl1param::T*> trlwedevices;
    cuFHETRLWElvl1();
    ~cuFHETRLWElvl1();

   private:
    // Don't allow users to copy this struct.
    cuFHETRLWElvl1(const cuFHETRLWElvl1&);
    cuFHETRLWElvl1& operator=(const cuFHETRLWElvl1&);
};

struct cuFHETRGSWNTTlvl1{
    TFHEpp::TRGSWNTT<TFHEpp::lvl1param> trgswhost;
    std::vector<FFP*> trgswdevices;
    cuFHETRGSWNTTlvl1();
    ~cuFHETRGSWNTTlvl1();

   private:
    // Don't allow users to copy this struct.
    cuFHETRGSWNTTlvl1(const cuFHETRGSWNTTlvl1&);
    cuFHETRGSWNTTlvl1& operator=(const cuFHETRGSWNTTlvl1&);
};

}  // namespace cufhe
