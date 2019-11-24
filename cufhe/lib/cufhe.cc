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
#include <include/details/allocator_cpu.h>

#include <math.h>
#include <random>
#include <iostream>

namespace cufhe {

Param kParam = {
  DEF_n, //n
  DEF_N, //N
  DEF_k, //k
  DEF_Bgbit, //Bgbit
  DEF_l, //l
  DEF_basebit, //basebit
  DEF_t, //t
  DEF_alpha, //alpha
  DEF_bkalpha //bkalpha
};

Param* GetDefaultParam() { return &kParam; }

////////////////////////////////////////////////////////////////////////////////
std::default_random_engine generator; // @todo Set Seed!

void RandomGeneratorSetSeed(uint32_t* values, int32_t size = 1) {
    std::seed_seq seeds(values, values + size);
    generator.seed(seeds);
}

inline
double SDFromBound(const double& noise_bound) {
  return noise_bound * sqrt(2.0 / M_PI);
}

// Conversions go back to -0.5~0.5 if Torus is int32_t!!!
inline
Torus TorusFromDouble(double d) {
  return Torus(int64_t((d - int64_t(d)) * (INT64_C(1) << 32)));
}

inline
double DoubleFromTorus(Torus t) {
  return double(t) / (double)(INT64_C(1) << 32);
}

Torus ApproxPhase(Torus phase, int32_t msg_space) {
    uint64_t interv = (UINT64_C(1) << 63) / msg_space * 2;
    uint64_t half_interval = interv / 2;
    uint64_t phase64 = (uint64_t(phase) << 32) + half_interval;
    //floor to the nearest multiples of interv
    phase64 -= phase64 % interv;
    //rescale to torus32
    return Torus(phase64 >> 32);
}


template <typename Torus>
void PolyMulAddBinary(Torus *r, Torus *a, Binary *b, uint32_t n) {
  // @todo Use CUDA if necessary.
/*
  for (int i = 0; i < n; i ++) {
    for (int j = 0; j < n - i; j ++)
      r[i + j] += a[i] * b[j];
    for (int j = n - i; j < n; j ++)
      r[i + j - n] -= a[i] * b[j];
  }
*/
  for (int i = 0; i < n; i ++) {
    for (int j = 0; j < n - i; j ++)
      r[i + j] += a[i] & (-b[j]);
    for (int j = n - i; j < n; j ++)
      r[i + j - n] -= a[i] & (-b[j]);
  }
}

////////////////////////////////////////////////////////////////////////////////
void LWEKeyGen(LWEKey* key, const Param* param = GetDefaultParam()) {
  std::uniform_int_distribution<> dist(0, 1);
  for (int i = 0; i < key->n(); i ++)
    key->data()[i] = dist(generator);
}

void TLWEKeyGen(TLWEKey* key, const Param* param = GetDefaultParam()) {
  std::uniform_int_distribution<> dist(0, 1);
  for (int i = 0; i < key->NumPolys(); i ++)
    for (int j = 0; j < key->n(); j ++)
      key->ExtractPoly(i)[j] = dist(generator);
}

void LWEEncrypt(LWESample* ct, const Torus pt, const LWEKey* key,
                const double noise_bound = GetDefaultParam()->lwe_noise_) {
  std::normal_distribution<double> dist_b(0.0, SDFromBound(noise_bound));
  ct->b() = pt + TorusFromDouble(dist_b(generator));
  std::uniform_int_distribution<Torus> dist_a(
      std::numeric_limits<Torus>::min(), std::numeric_limits<Torus>::max());
  for (int i = 0; i < key->n(); i ++) {
    ct->a()[i] = dist_a(generator);
    ct->b() += ct->a()[i] * key->data()[i];
  }
}

void LWEEncryptExternalNoise(LWESample* ct, const Torus pt, const LWEKey* key,
                const double noise) {
  ct->b() = pt + TorusFromDouble(noise);
  std::uniform_int_distribution<Torus> dist_a(
      std::numeric_limits<Torus>::min(), std::numeric_limits<Torus>::max());
//  std::uniform_int_distribution<Torus> dist_a(INT32_MIN, INT32_MAX);
  for (int i = 0; i < key->n(); i ++) {
    ct->a()[i] = dist_a(generator);
    ct->b() += ct->a()[i] * key->data()[i];
  }
}

void LWEDecrypt(Torus& pt, const LWESample* ct, const LWEKey* key,
                uint32_t space) {
  assert(ct->n() == key->n());
  Torus err = ct->b();
  for (int i = 0; i < ct->n(); i ++)
    err -= ct->a()[i] * key->data()[i];
  pt = err;//pt = ApproxPhase(err, space);
}

void KeySwitchingKeyGen(KeySwitchingKey* key,
                        const LWEKey* lwe_key_to,
                        const LWEKey* lwe_key_from,
                        const Param* param = GetDefaultParam()) {
  Torus mu;
  uint32_t temp;
  LWESample* lwe_sample = new LWESample;
  const uint32_t total = key->NumLWESamples();
  double* noise = new double[total];
  double error = 0;
  for (int i = 0; i < total; i ++) {
    std::normal_distribution<double> dist(0.0, SDFromBound(param->lwe_noise_));
    noise[i] = dist(generator);
    error += noise[i];
  }
  error /= total;
  for (int i = 0; i < total; i ++)
    noise[i] -= error;

  uint32_t index = 0;
  for (int i = 0; i < key->m(); i ++) {
    temp = lwe_key_from->data()[i];
    for (int j = 0; j < key->l(); j ++) {
      for (int k = 0; k < (0x1 << key->w()); k ++) {
        key->ExtractLWESample(lwe_sample, key->GetLWESampleIndex(i, j, k));
        mu = (temp * k) * (1 << (32 - (j + 1) * key->w()));
        LWEEncryptExternalNoise(lwe_sample, mu, lwe_key_to, noise[index]);
        index ++;
      }
    }
  }
  delete lwe_sample;
  delete [] noise;
}

void TLWEEncryptZero(TLWESample* ct, const TLWEKey* key,
                     const double noise_bound = GetDefaultParam()->tlwe_noise_) {
  std::normal_distribution<double> dist_b(0.0, SDFromBound(noise_bound));
  for (int i = 0; i < key->n(); i ++)
    ct->b()[i] = TorusFromDouble(dist_b(generator));
  std::uniform_int_distribution<Torus> dist_a(
      std::numeric_limits<Torus>::min(), std::numeric_limits<Torus>::max());
  for (int i = 0; i < key->k(); i ++) {
    for (int j = 0; j < key->n(); j ++)
      ct->a(i)[j] = dist_a(generator);
    PolyMulAddBinary<Torus>(ct->b(), ct->a(i), key->data(), key->n());
  }
}

void TGSWEncryptBinary(TGSWSample* ct, const Binary pt, const TGSWKey* key,
                       const Param* param = GetDefaultParam()) {
  uint32_t l = ct->l();
  uint32_t k = ct->k();
  uint32_t w = ct->w();
  TLWESample* tlwe_sample = new TLWESample;
  for (int i = 0; i < ct->NumTLWESamples(); i ++) {
    ct->ExtractTLWESample(tlwe_sample, i);
    TLWEEncryptZero(tlwe_sample, key);
  }
  for (int i = 0; i < l; i ++) {
    Torus mu = (Torus)pt << (32 - w * (i + 1));
    for (int j = 0; j < k; j ++) {
      ct->ExtractTLWESample(tlwe_sample, j * l + i);
      tlwe_sample->a(j)[0] += mu;
    }
    ct->ExtractTLWESample(tlwe_sample, k * l + i);
    tlwe_sample->b()[0] += mu;
  }
  delete tlwe_sample;
}

void BootstrappingKeyGen(BootstrappingKey* key,
                         const LWEKey* lwe_key,
                         const TGSWKey* tgsw_key,
                         const Param* param = GetDefaultParam()) {
  TGSWSample* tgsw_sample = new TGSWSample;
  for (int i = 0; i < lwe_key->n(); i ++) {
    key->ExtractTGSWSample(tgsw_sample, i);
    TGSWEncryptBinary(tgsw_sample, lwe_key->data()[i], tgsw_key);
  }
  delete tgsw_sample;
}

////////////////////////////////////////////////////////////////////////////////

PriKey::PriKey(bool is_alias) {
  Param* param = GetDefaultParam();
  lwe_key_ = new LWEKey(param->lwe_n_);
  tlwe_key_ = new TLWEKey(param->tlwe_n_, param->tlwe_k_);
  lwe_key_deleter_ = nullptr;
  tlwe_key_deleter_ = nullptr;
  std::pair<void*, MemoryDeleter> pair;
  pair = AllocatorCPU::New(lwe_key_->SizeMalloc());
  lwe_key_->set_data((LWEKey::PointerType)pair.first);
  lwe_key_deleter_ = pair.second;
  pair = AllocatorCPU::New(tlwe_key_->SizeMalloc());
  tlwe_key_->set_data((TLWEKey::PointerType)pair.first);
  tlwe_key_deleter_ = pair.second;
}

PriKey::~PriKey() {
  lwe_key_deleter_(lwe_key_->data());
  tlwe_key_deleter_(tlwe_key_->data());
  delete lwe_key_;
  delete tlwe_key_;
}

PubKey::PubKey(bool is_alias) {
  Param* param = GetDefaultParam();
  bk_ = new BootstrappingKey(param->tlwe_n_, param->tlwe_k_,
                             param->tgsw_decomp_size_,
                             param->tgsw_decomp_bits_, param->lwe_n_);
  ksk_ = new KeySwitchingKey(param->lwe_n_,
                             param->keyswitching_decomp_size_,
                             param->keyswitching_decomp_bits_,
                             param->tlwe_n_ * param->tlwe_k_);
  bk_deleter_ = nullptr;
  ksk_deleter_ = nullptr;
  std::pair<void*, MemoryDeleter> pair;
  pair = AllocatorCPU::New(bk_->SizeMalloc());
  bk_->set_data((BootstrappingKey::PointerType)pair.first);
  bk_deleter_ = pair.second;
  pair = AllocatorCPU::New(ksk_->SizeMalloc());
  ksk_->set_data((KeySwitchingKey::PointerType)pair.first);
  ksk_deleter_ = pair.second;
}

PubKey::~PubKey() {
  bk_deleter_(bk_->data());
  ksk_deleter_(ksk_->data());
  delete bk_;
  delete ksk_;
}

Ctxt::~Ctxt() {
  if(lwe_sample_ != nullptr) {
    if(lwe_sample_deleter_ != nullptr) {
      lwe_sample_deleter_(lwe_sample_->data());
      lwe_sample_deleter_ = nullptr;
    }

    lwe_sample_->set_data(nullptr);
    delete lwe_sample_;
    lwe_sample_ = nullptr;
  }

  if(lwe_sample_device_ != nullptr) {
    if(lwe_sample_device_deleter_ != nullptr) {
      lwe_sample_device_deleter_(lwe_sample_device_->data());
      lwe_sample_device_deleter_ = nullptr;
    }

    lwe_sample_device_->set_data(nullptr);
    delete lwe_sample_device_;
    lwe_sample_device_ = nullptr;
  }
}

void Ctxt::assign(void* host_ptr, void* device_ptr) {
  if(lwe_sample_deleter_ != nullptr) {
    lwe_sample_deleter_(lwe_sample_->data());
    lwe_sample_deleter_ = nullptr;
  }

  lwe_sample_->set_data((LWESample::PointerType)host_ptr);

  if(lwe_sample_device_deleter_ != nullptr) {
    lwe_sample_device_deleter_(lwe_sample_device_->data());
    lwe_sample_device_deleter_ = nullptr;
  }

  lwe_sample_device_->set_data((LWESample::PointerType)device_ptr);
}

void SetSeed(uint32_t seed) {
  srand(seed);
  RandomGeneratorSetSeed(&seed, 1);
}

void PubKeyGen(PubKey& pub_key, const PriKey& pri_key) {
  BootstrappingKeyGen(pub_key.bk_, pri_key.lwe_key_, pri_key.tlwe_key_);
  LWEKey* lwe_key_extract = new LWEKey;
  pri_key.tlwe_key_->ExtractLWEKey(lwe_key_extract);
  KeySwitchingKeyGen(pub_key.ksk_,
                     pri_key.lwe_key_,
                     lwe_key_extract);
  delete lwe_key_extract;
}

void PriKeyGen(PriKey& pri_key) {
  LWEKeyGen(pri_key.lwe_key_);
  TLWEKeyGen(pri_key.tlwe_key_);
}

void KeyGen(PubKey& pub_key, PriKey& pri_key) {
  PriKeyGen(pri_key);
  PubKeyGen(pub_key, pri_key);
}

void Encrypt(Ctxt& ctxt, const Ptxt& ptxt, const PriKey& pri_key) {
  //assert(Find(ptxt, kPtxtSet));
//  Torus mu = TorusFromDouble((double)1.0 * ptxt.message_ / Ptxt::kPtxtSpace);
  Torus one = ModSwitchToTorus(1, 8);
  Torus mu = ptxt.message_ ? one : -one;
  LWEEncrypt(ctxt.lwe_sample_, mu, pri_key.lwe_key_);
}

void Decrypt(Ptxt& ptxt, const Ctxt& ctxt, const PriKey& pri_key) {
  Torus mu;
  LWEDecrypt(mu, ctxt.lwe_sample_, pri_key.lwe_key_, Ptxt::kPtxtSpace);
//  ptxt.message_ = (uint32_t)int32_t((Ptxt::kPtxtSpace) * DoubleFromTorus(mu));
//  ptxt.message_ %= ptxt.kPtxtSpace;
  ptxt.message_ = mu > 0 ? 1 : 0;
}

} // namespace cufhe
