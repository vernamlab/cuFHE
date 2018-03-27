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

#include "cufhe_core.h"
#include "details/allocator.h"
#include <iostream>
#include <math.h>
#include <time.h>

namespace cufhe {

/*****************************
 * Macros *
 *****************************/

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
  double   lwe_noise_;
  double   tlwe_noise_;

  Param() :
      lwe_n_(500), tlwe_n_(1024), tlwe_k_(1), tgsw_decomp_bits_(10),
      tgsw_decomp_size_(2), keyswitching_decomp_bits_(2),
      keyswitching_decomp_size_(8), lwe_noise_(pow(2.0, -15)),
      tlwe_noise_(9.e-9) {};

  Param(uint32_t lwe_n, uint32_t tlwe_n, uint32_t tlwe_k,
        uint32_t tgsw_decomp_bits, uint32_t tgsw_decomp_size,
	      uint32_t keyswitching_decomp_bits, uint32_t keyswitching_decomp_size,
	      double lwe_noise, double tlwe_noise) :
      lwe_n_(lwe_n), tlwe_n_(tlwe_n), tlwe_k_(tlwe_k),
	    tgsw_decomp_bits_(tgsw_decomp_bits),
      tgsw_decomp_size_(tgsw_decomp_size),
      keyswitching_decomp_bits_(keyswitching_decomp_bits),
      keyswitching_decomp_size_(keyswitching_decomp_size),
      lwe_noise_(lwe_noise), tlwe_noise_(tlwe_noise) {};
};

Param* GetDefaultParam();

/**
 * Private Key.
 * Necessary for encryption/decryption and public key generation.
 */
struct PriKey {
  PriKey(bool is_alias = false);
  ~PriKey();
  LWEKey* lwe_key_;
  TLWEKey* tlwe_key_;
  MemoryDeleter lwe_key_deleter_;
  MemoryDeleter tlwe_key_deleter_;
};

/**
 * Public Key.
 * Necessary for a server to perform homomorphic evaluation.
 */
struct PubKey {
  PubKey(bool is_alias = false);
  ~PubKey();
  BootstrappingKey* bk_;
  KeySwitchingKey* ksk_;
  MemoryDeleter bk_deleter_;
  MemoryDeleter ksk_deleter_;
};

/** Ciphertext. */
struct Ctxt {
  Ctxt(bool is_alias = false);
  ~Ctxt();
  LWESample* lwe_sample_;
  MemoryDeleter lwe_sample_deleter_;
};

/** Plaintext is in {0, 1}. */
struct Ptxt {
  inline Ptxt& operator=(uint32_t message) {
    this->message_ = message % kPtxtSpace;
    return *this;
  }
  uint32_t message_;
  static const uint32_t kPtxtSpace = 2;

  void set(uint32_t message) { message_ = message % kPtxtSpace; };
  uint32_t get() { return message_; }
};

/******************
 * Client Methods *
 ******************/
void SetSeed(uint32_t seed = time(nullptr));
void PriKeyGen(PriKey& pri_key);
void PubKeyGen(PubKey& pub_key, const PriKey& pri_key);
void KeyGen(PubKey& pub_key, PriKey& pri_key);
void Encrypt(Ctxt& ctxt, const Ptxt& ptxt, const PriKey& pri_key);
void Decrypt(Ptxt& ptxt, const Ctxt& ctxt, const PriKey& pri_key);

/******************
 * I/O Methods *
 ******************/
// not ready
typedef std::string FileName;
void WritePriKeyToFile(const PriKey& pri_key, FileName file);
void ReadPriKeyFromFile(PriKey& pri_key, FileName file);
void WritePubKeyToFile(const PubKey& pub_key, FileName file);
void ReadPubKeyFromFile(PubKey& pub_key, FileName file);
void WriteCtxtToFile(const Ctxt& ct, FileName file);
void ReadCtxtFromFile(Ctxt& ct, FileName file);

} // namespace cufhe
