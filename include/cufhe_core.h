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

#include <stdint.h>

#include "details/assert.h"
#include "details/math.h"

namespace cufhe {

using Torus = int32_t;
using Binary = uint32_t;

template <typename T>
class DataTemplate {
   public:
    typedef T DataType;
    typedef T* PointerType;

    __host__ __device__ DataTemplate(T* data = nullptr) : data_(data) {}

    __host__ __device__ virtual ~DataTemplate() {}

    __host__ __device__ inline size_t SizeMalloc() const
    {
        return sizeof(T) * Align512(SizeData() / sizeof(T));
    }

    __host__ __device__ inline virtual size_t SizeData() const { return 0; }

    __host__ __device__ inline T* data() const { return data_; }

    __host__ __device__ inline void set_data(T* data) { data_ = data; }

   private:
    T* data_;
};  // class DataTemplate

template <typename T = Torus>
class LWESample_T : public DataTemplate<T> {
   public:
    __host__ __device__ LWESample_T() : n_(0) {}
    __host__ __device__ LWESample_T(uint32_t n, T* data = nullptr)
        : DataTemplate<T>(data), n_(n)
    {
    }
    __host__ __device__ ~LWESample_T() {}
    __host__ __device__ inline size_t SizeData() const override
    {
        return sizeof(T) * (this->n_ + 1);
    }
    __host__ __device__ inline T* a() const { return this->data(); }
    __host__ __device__ inline T& b() const { return this->data()[n_]; }
    __host__ __device__ inline uint32_t n() const { return this->n_; }
    __host__ __device__ inline void set_n(uint32_t n) { n_ = n; }

   private:
    uint32_t n_;
};  // class LWESample_T

template <typename T = Binary>
class LWEKey_T : public DataTemplate<T> {
   public:
    __host__ __device__ LWEKey_T() : n_(0) {}
    __host__ __device__ LWEKey_T(uint32_t n, T* data = nullptr)
        : DataTemplate<T>(data), n_(n)
    {
    }
    __host__ __device__ ~LWEKey_T() {}
    __host__ __device__ inline size_t SizeData() const override
    {
        return sizeof(T) * this->n_;
    }
    __host__ __device__ inline uint32_t n() const { return this->n_; }
    __host__ __device__ inline void set_n(uint32_t n) { n_ = n; }

   private:
    uint32_t n_;
};  // class LWEKey_T

template <typename T = Torus>
class TLWESample_T : public DataTemplate<T> {
   public:
    __host__ __device__ TLWESample_T() : n_(0), k_(0) {}
    __host__ __device__ TLWESample_T(uint32_t n, uint32_t k, T* data = nullptr)
        : n_(n), k_(k), DataTemplate<T>(data)
    {
        Assert(n == Pow2(Log2(n)));
        Assert(k >= 1);
    }
    __host__ __device__ ~TLWESample_T() {}
    __host__ __device__ inline size_t SizeData() const override
    {
        return sizeof(T) * this->n_ * (this->k_ + 1);
    }
    __host__ __device__ inline uint32_t n() const { return this->n_; }
    __host__ __device__ inline uint32_t k() const { return this->k_; }
    __host__ __device__ inline void set_n(uint32_t n) { n_ = n; }
    __host__ __device__ inline void set_k(uint32_t k) { k_ = k; }
    __host__ __device__ inline uint32_t NumPolys() const
    {
        return this->k_ + 1;
    }
    __host__ __device__ inline T* ExtractPoly(uint32_t index) const
    {
        Assert(index <= this->k_);
        return this->data() + index * this->n_;
    }
    __host__ __device__ inline T* a(uint32_t index = 0) const
    {
        Assert(index < this->k_);
        return this->ExtractPoly(index);
    }
    __host__ __device__ inline T* b() const { return ExtractPoly(this->k_); }

   private:
    uint32_t n_;
    uint32_t k_;
};  // class TLWESample_T

template <typename T = Binary>
class TLWEKey_T : public DataTemplate<T> {
   public:
    __host__ __device__ TLWEKey_T() : n_(0), k_(0) {}
    __host__ __device__ TLWEKey_T(uint32_t n, uint32_t k, T* data = nullptr)
        : n_(n), k_(k), DataTemplate<T>(data)
    {
        Assert(n == Pow2(Log2(n)));
        Assert(k >= 1);
    }
    __host__ __device__ ~TLWEKey_T() {}
    __host__ __device__ inline size_t SizeData() const override
    {
        return sizeof(T) * this->n_ * (this->k_ + 1);
    }
    __host__ __device__ inline uint32_t n() const { return this->n_; }
    __host__ __device__ inline uint32_t k() const { return this->k_; }
    __host__ __device__ inline void set_n(uint32_t n) { n_ = n; }
    __host__ __device__ inline void set_k(uint32_t k) { k_ = k; }
    __host__ __device__ inline uint32_t NumPolys() const { return this->k_; }
    __host__ __device__ inline T* ExtractPoly(uint32_t index) const
    {
        Assert(index < this->k_);
        return this->data() + index * this->n_;
    }

    __host__ __device__ inline void ExtractLWEKey(LWEKey_T<T>* out) const
    {
        out->set_n(this->n_ * this->k_);
        out->set_data(this->data());
    }

    __host__ __device__ inline LWEKey_T<T> ExtractLWEKey() const
    {
        LWEKey_T<T> out;
        out.set_n(this->n_ * this->k_);
        out.set_data(this->data());
        return out;
    }

   private:
    uint32_t n_;
    uint32_t k_;
};  // class TLWEKey_T

template <typename T = Torus>
class TGSWSample_T : public DataTemplate<T> {
   public:
    __host__ __device__ TGSWSample_T() : n_(0), k_(0), l_(0), w_(0) {}
    __host__ __device__ TGSWSample_T(uint32_t n, uint32_t k, uint32_t l,
                                     uint32_t w, T* data = nullptr)
        : n_(n), k_(k), l_(l), w_(w), DataTemplate<T>(data)
    {
        Assert(n == Pow2(Log2(n)));
        Assert(k >= 1);
        Assert(l * w <= 8 * sizeof(T));
    }
    __host__ __device__ ~TGSWSample_T() {}
    __host__ __device__ inline size_t SizeData() const override
    {
        return TLWESample_T<T>(this->n_, this->k_).SizeMalloc() *
               this->NumTLWESamples();
    }
    __host__ __device__ inline uint32_t n() const { return this->n_; }
    __host__ __device__ inline uint32_t k() const { return this->k_; }
    __host__ __device__ inline uint32_t l() const { return this->l_; }
    __host__ __device__ inline uint32_t w() const { return this->w_; }
    __host__ __device__ inline void set_n(uint32_t n) { n_ = n; }
    __host__ __device__ inline void set_k(uint32_t k) { k_ = k; }
    __host__ __device__ inline void set_l(uint32_t l) { l_ = l; }
    __host__ __device__ inline void set_w(uint32_t w) { w_ = w; }
    __host__ __device__ inline uint32_t NumTLWESamples() const
    {
        return (this->k_ + 1) * this->l_;
    }
    __host__ __device__ inline void ExtractTLWESample(TLWESample_T<T>* out,
                                                      uint32_t index) const
    {
        Assert(index < NumTLWESamples());
        out->set_n(this->n_);
        out->set_k(this->k_);
        out->set_data(this->data() + out->SizeMalloc() / sizeof(T) * index);
    }
    __host__ __device__ inline TLWESample_T<T> ExtractTLWESample(
        uint32_t index) const
    {
        TLWESample_T<T> out;
        Assert(index < NumTLWESamples());
        out.set_n(this->n_);
        out.set_k(this->k_);
        out.set_data(this->data() + out.SizeMalloc() / sizeof(T) * index);
        return out;
    }

   private:
    uint32_t n_;
    uint32_t k_;
    uint32_t l_;
    uint32_t w_;
};  // class TGSWSample_T

template <typename T = Torus>
class TGSWSampleArray_T : public DataTemplate<T> {
   public:
    __host__ __device__ TGSWSampleArray_T() : n_(0), k_(0), l_(0), w_(0), t_(0)
    {
    }
    __host__ __device__ TGSWSampleArray_T(uint32_t n, uint32_t k, uint32_t l,
                                          uint32_t w, uint32_t t)
        : n_(n), k_(k), l_(l), w_(w), t_(t)
    {
        Assert(n == Pow2(Log2(n)));
        Assert(k >= 1);
        Assert(l * w <= 8 * sizeof(T));
        Assert(t >= 1);
    }
    __host__ __device__ ~TGSWSampleArray_T() {}
    __host__ __device__ inline size_t SizeData() const override
    {
        return TGSWSample_T<T>(this->n_, this->k_, this->l_, this->w_)
                   .SizeMalloc() *
               this->NumTGSWSamples();
    }
    __host__ __device__ inline uint32_t n() const { return this->n_; }
    __host__ __device__ inline uint32_t k() const { return this->k_; }
    __host__ __device__ inline uint32_t l() const { return this->l_; }
    __host__ __device__ inline uint32_t w() const { return this->w_; }
    __host__ __device__ inline uint32_t t() const { return this->t_; }
    __host__ __device__ inline void set_n(uint32_t n) { n_ = n; }
    __host__ __device__ inline void set_k(uint32_t k) { k_ = k; }
    __host__ __device__ inline void set_l(uint32_t l) { l_ = l; }
    __host__ __device__ inline void set_w(uint32_t w) { w_ = w; }
    __host__ __device__ inline void set_t(uint32_t t) { t_ = t; }
    __host__ __device__ inline uint32_t NumTGSWSamples() const
    {
        return this->t_;
    }
    __host__ __device__ inline void ExtractTGSWSample(TGSWSample_T<T>* out,
                                                      uint32_t index) const
    {
        Assert(index < NumTGSWSamples());
        out->set_n(this->n_);
        out->set_k(this->k_);
        out->set_l(this->l_);
        out->set_w(this->w_);
        out->set_data(this->data() + out->SizeMalloc() / sizeof(T) * index);
    }
    __host__ __device__ inline TGSWSample_T<T> ExtractTGSWSample(
        uint32_t index) const
    {
        TGSWSample_T<T> out;
        Assert(index < NumTGSWSamples());
        out.set_n(this->n_);
        out.set_k(this->k_);
        out.set_l(this->l_);
        out.set_w(this->w_);
        out.set_data(this->data() + out.SizeMalloc() / sizeof(T) * index);
        return out;
    }

   private:
    uint32_t n_;
    uint32_t k_;
    uint32_t l_;
    uint32_t w_;
    uint32_t t_;
};  // class TGSWSampleArray_T

template <typename T = Torus>
class LWESampleArray_T : public DataTemplate<T> {
   public:
    __host__ __device__ LWESampleArray_T() : n_(0), t_(0) {}
    __host__ __device__ LWESampleArray_T(uint32_t n, uint32_t t) : n_(n), t_(t)
    {
        Assert(t >= 1);
    }
    __host__ __device__ ~LWESampleArray_T() {}
    __host__ __device__ inline size_t SizeData() const override
    {
        // use SizeMalloc !
        return LWESample_T<T>(this->n_).SizeMalloc() * this->t_;
    }
    __host__ __device__ inline uint32_t n() const { return this->n_; }
    __host__ __device__ inline uint32_t t() const { return this->t_; }
    __host__ __device__ inline void set_n(uint32_t n) { n_ = n; }
    __host__ __device__ inline void set_t(uint32_t t) { t_ = t; }
    __host__ __device__ inline uint32_t NumLWESamples() const
    {
        return this->t_;
    }
    __host__ __device__ inline void ExtractLWESample(LWESample_T<T>* out,
                                                     uint32_t index) const
    {
        Assert(index < NumLWESamples());
        out->set_n(n_);
        out->set_data(this->data() + out->SizeMalloc() / sizeof(T) * index);
    }
    __host__ __device__ inline LWESample_T<T> ExtractLWESample(
        uint32_t index) const
    {
        LWESample_T<T> out;
        Assert(index < NumLWESamples());
        out.set_n(n_);
        out.set_data(this->data() + out.SizeMalloc() / sizeof(T) * index);
        return out;
    }

   private:
    uint32_t n_;
    uint32_t t_;
};  // class LWESampleArray_T

template <typename T = Torus>
class KeySwitchingKey_T : public LWESampleArray_T<T> {
   public:
    __host__ __device__ KeySwitchingKey_T() : l_(0), w_(0), m_(0) {}
    __host__ __device__ KeySwitchingKey_T(uint32_t n, uint32_t l, uint32_t w,
                                          uint32_t m)
        : LWESampleArray_T<T>(n, m * l << w), l_(l), w_(w), m_(m)
    {
        Assert(l * w <= 8 * sizeof(T));
        Assert(m >= 1);
    }
    __host__ __device__ ~KeySwitchingKey_T() {}
    __host__ __device__ uint32_t GetLWESampleIndex(uint32_t degree,
                                                   uint32_t digit,
                                                   uint32_t value)
    {
        Assert(degree < this->m_);
        Assert(digit < this->l_);
        Assert(value < (0x1 << this->w_));
        return ((degree * this->l_ + digit) << this->w_) + value;
    }
    __host__ __device__ inline uint32_t l() const { return this->l_; }
    __host__ __device__ inline uint32_t w() const { return this->w_; }
    __host__ __device__ inline uint32_t m() const { return this->m_; }
    __host__ __device__ inline void set_l(uint32_t l) { l_ = l; }
    __host__ __device__ inline void set_w(uint32_t w) { w_ = w; }
    __host__ __device__ inline void set_m(uint32_t m) { m_ = m; }

   private:
    uint32_t l_;
    uint32_t w_;
    uint32_t m_;
};  // class KeySwitchingKey_T

using LWESample = LWESample_T<Torus>;
using LWEKey = LWEKey_T<Binary>;
using TLWESample = TLWESample_T<Torus>;
using TLWEKey = TLWEKey_T<Binary>;
using TGSWSample = TGSWSample_T<Torus>;
using TGSWKey = TLWEKey;
using BootstrappingKey = TGSWSampleArray_T<Torus>;
using KeySwitchingKey = KeySwitchingKey_T<Torus>;

inline Torus ModSwitchToTorus(int32_t mu, int32_t space)
{
    uint64_t gap = ((UINT64_C(1) << 63) / space) * 2;
    return int32_t((uint64_t)(mu * gap) >> 32);
}

}  // namespace cufhe
