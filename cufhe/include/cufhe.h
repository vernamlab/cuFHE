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
};

/**
 * Private Key.
 * Necessary for encryption/decryption and public key generation.
 */
struct PriKey {
  PriKey(bool is_alias = false);
  ~PriKey();

  template <typename AllocatorClass>
  void New() {
    std::pair<void*, MemoryDeleter> pair;
    pair = AllocatorClass::New(lwe_key_->SizeMalloc());
    lwe_key_->set_data((LWEKey::PointerType)pair.first);
    lwe_key_deleter_ = pair.second;
    pair = AllocatorClass::New(tlwe_key_->SizeMalloc());
    tlwe_key_->set_data((TLWEKey::PointerType)pair.first);
    tlwe_key_deleter_ = pair.second;
  }

  void Delete();

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

  template <typename AllocatorClass>
  void New() {
    std::pair<void*, MemoryDeleter> pair;
    pair = AllocatorClass::New(bk_->SizeMalloc());
    bk_->set_data((BootstrappingKey::PointerType)pair.first);
    bk_deleter_ = pair.second;
    pair = AllocatorClass::New(ksk_->SizeMalloc());
    ksk_->set_data((KeySwitchingKey::PointerType)pair.first);
    ksk_deleter_ = pair.second;
  }

  void Delete();

  BootstrappingKey* bk_;
  KeySwitchingKey* ksk_;
  MemoryDeleter bk_deleter_;
  MemoryDeleter ksk_deleter_;
};

/** Ciphertext. */
struct Ctxt {
  Ctxt(bool is_alias = false);
  ~Ctxt();

  template <typename AllocatorClass>
  void New() {
    std::pair<void*, MemoryDeleter> pair;
    pair = AllocatorClass::New(lwe_sample_->SizeMalloc());
    lwe_sample_->set_data((LWESample::PointerType)pair.first);
    lwe_sample_deleter_ = pair.second;
  }

  void Delete();

  LWESample* lwe_sample_;
  MemoryDeleter lwe_sample_deleter_;
};

/** Plaintext is in {0, 1}. */
struct Ptxt {
  uint32_t message_;
  static const uint32_t kPtxtSpace = 2;
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
