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

#include <params.hpp>
#include "details/allocator.h"

namespace cufhe {
class FFP;

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
