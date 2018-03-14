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

#pragma once

#include <stdint.h>

namespace cufhe {

/**
 * @class FFP
 * @brief Wraps a uint64_t integer as an element in the field FF(P). And defines
 *        arithmetic operations (operations). P = 2^64-2^32+1.
 * @details Do not use function Pow(...) in performance-critical code.
 * @details All function members are inline.
 * @details Consider all arithmetic functions for device only.
 */

class FFP {

private:

  /** Field modulus P. */
  static const uint64_t kModulus_ = 0xffffffff00000001UL;

  /** A 2^32-th primitive root of unity mod P. */
  static const uint64_t kRoot2e32_ = 0xa70dc47e4cbdf43fUL;

  /** An 64-bit unsigned integer within [0, P-1]. */
  uint64_t val_;

  /** A 64-bit integer modulo P. */
  /** Cannot avoid divergence, since comparison causes branches in CUDA. */
  __device__ inline
  void ModP(uint64_t& x) {
    asm("{\n\t"
        ".reg .u32        m;\n\t"
        ".reg .u64        t;\n\t"
        "set.ge.u32.u64   m, %0, %1;\n\t"
        "mov.b64          t, {m, 0};\n\t"
        "add.u64         %0, %0, t;\n\t"
        "}"
        : "+l"(x)
        : "l"(kModulus_));
  }

public:

  //////////////////////////////////////////////////////////////////////////////
  // Access and set value

  /** Default constructor. Value is not specified. */
  __host__ __device__ inline
  FFP() {} // shared memory requires empty constructor

  /**
   * Setting value to a (mod P).
   * @param[in] a An unsigned integer in [0, P-1].
   *            Immediates are uint32_t unless specified.
   */
  __host__ __device__ inline
  FFP(uint8_t a) { val_ = a; };

  __host__ __device__ inline
  FFP(uint16_t a) { val_ = a; };

  __host__ __device__ inline
  FFP(uint32_t a) { val_ = a; };

  __host__ __device__ inline
  FFP(uint64_t a) { val_ = a; };

  __host__ __device__ inline
  FFP(int8_t a) { val_ = (uint64_t)a - (uint32_t)(-(a < 0)); };

  __host__ __device__ inline
  FFP(int16_t a) { val_ = (uint64_t)a - (uint32_t)(-(a < 0)); };

  __host__ __device__ inline
  FFP(int32_t a) { val_ = (uint64_t)a - (uint32_t)(-(a < 0)); };

  __host__ __device__ inline
  FFP(int64_t a) { val_ = (uint64_t)a - (uint32_t)(-(a < 0)); };

  /** Default destructor. Value is not wiped. */
  __host__ __device__ inline
  ~FFP() {}

  /** Get value. */
  __host__ __device__ inline
  uint64_t& val() { return val_; }

  /** Get value. */
  __host__ __device__ inline
  const uint64_t& val() const { return val_; }

  /** Return modulus P. */
  __host__ __device__ inline
  static uint64_t kModulus() { return kModulus_; };

  /** Return 2^32-th primitive root of unity mod P. */
  __host__ __device__ inline
  static uint64_t kRoot2e32() { return kRoot2e32_; };

  //////////////////////////////////////////////////////////////////////////////
  // Operators

  /**
   * Assign.
   * @param a [description]
   */
  __host__ __device__ inline
  FFP& operator=(uint8_t a) { this->val_ = a; return *this; };

  __host__ __device__ inline
  FFP& operator=(uint16_t a) { this->val_ = a; return *this; };

  __host__ __device__ inline
  FFP& operator=(uint32_t a) { this->val_ = a; return *this; };

  __host__ __device__ inline
  FFP& operator=(uint64_t a) { this->val_ = a; return *this; };

  __host__ __device__ inline
  FFP& operator=(int8_t a) {
    this->val_ = (uint64_t)a - (uint32_t)(-(a < 0));
    return *this;
  };

  __host__ __device__ inline
  FFP& operator=(int16_t a) {
    this->val_ = (uint64_t)a - (uint32_t)(-(a < 0));
    return *this;
  };

  __host__ __device__ inline
  FFP& operator=(int32_t a) {
    this->val_ = (uint64_t)a - (uint32_t)(-(a < 0));
    return *this;
  };

  __host__ __device__ inline
  FFP& operator=(int64_t a) {
    this->val_ = (uint64_t)a - (uint32_t)(-(a < 0));
    return *this;
  };

  __host__ __device__ inline
  FFP& operator=(FFP a) { this->val_ = a.val(); return *this; }

  /** Explicit conversion. */
  __host__ __device__ inline
  explicit operator uint64_t() { return val_; } // correct result

  __host__ __device__ inline
  explicit operator uint8_t() { return (uint8_t)val_; } // truncated result

  __host__ __device__ inline
  explicit operator uint16_t() { return (uint16_t)val_; } // truncated result

  __host__ __device__ inline
  explicit operator uint32_t() { return (uint32_t)val_; } // truncated result

  /** Addition in FF(P): val_ = val_ + a mod P. */
  __device__ inline
  FFP& operator+=(const FFP& a) { this->Add(*this, a); return *this; }

  /** Addition in FF(P): return a + b mod P. */
  friend __device__ inline
  FFP operator+(const FFP& a, const FFP& b) { FFP r; r.Add(a, b); return r; }

  /** Subtraction in FF(P): val_ = val_ - a mod P. */
  __device__ inline
  FFP& operator-=(const FFP& a) { this->Sub(*this, a); return *this; }

  /** Subtraction in FF(P): return a - b mod P. */
  friend __device__ inline
  FFP operator-(const FFP& a, const FFP& b) { FFP r; r.Sub(a, b); return r; }

  /** Multiplication in FF(P): val_ = val_ * a mod P. */
  __device__ inline
  FFP& operator*=(const FFP& a) { this->Mul(*this, a); return *this; }

  /** Multiplication in FF(P): return a * b mod P. */
  friend __device__ inline
  FFP operator*(const FFP& a, const FFP& b) { FFP r; r.Mul(a, b); return r; }

  /** Equality. */
  __host__ __device__ inline
  bool operator==(const FFP& other) { return (bool)(val_ == other.val()); }

  /** Inequality. */
  __host__ __device__ inline
  bool operator!=(const FFP& other) { return (bool)(val_ != other.val()); }

  //////////////////////////////////////////////////////////////////////////////
  // Miscellaneous

  /**
   * Return a primitive n-th root in FF(P): val_ ^ n = 1 mod P.
   * @param[in] n A power of 2.
   */
  __device__ inline
  static FFP Root(uint32_t n) {
    return Pow(kRoot2e32_, (uint32_t)((0x1UL << 32) / n));
  }

  /**
   * Return the inverse of 2^log_n in FF(P): 2^{-log_n} mod P.
   * @param log_n An integer in [0, 32]
   */
  __host__ __device__ inline
  static FFP InvPow2(uint32_t log_n) {
    uint32_t r[2];
    r[0] = (0x1 << (32 - log_n)) + 1;
    r[1] = -r[0];
    return FFP(*(uint64_t*)r);
  }

  /** Exchange values with a. */
  __host__ __device__ inline
  void Swap(FFP& a) {
    uint64_t t = val_;
    val_ = a.val_;
    a.val_ = t;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Arithmetic

  /** Addition in FF(P): val_ = a + b mod P. */
  __device__ inline
  void Add(const FFP& a, const FFP& b) {
    asm("{\n\t"
        ".reg .u32          m;\n\t"
        ".reg .u64          t;\n\t"
        ".reg .pred         p;\n\t"
        // this = a + b;
        "add.u64            %0, %1, %2;\n\t"
        // this += (uint32_t)(-(this < b || this >= FFP_MODULUS));
        "setp.lt.u64        p, %0, %2;\n\t"
        "set.ge.or.u32.u64  m, %0, %3, p;\n\t"
        "mov.b64            t, {m, 0};\n\t"
        "add.u64            %0, %0, t;\n\t"
        "}"
        : "+l"(val_)
        : "l"(a.val_), "l"(b.val_), "l"(kModulus_));
  }

  /** Subtraction in FF(P): val_ = a + b mod P. */
  __device__ inline
  void Sub(const FFP& a, const FFP& b) {
    register uint64_t r = 0;
    asm("{\n\t"
        ".reg .u32          m;\n\t"
        ".reg .u64          t;\n\t"
        // this = a - b;
        "sub.u64            %0, %1, %2;\n\t"
        // this -= (uint32_t)(-(this > a));
        "set.gt.u32.u64     m, %0, %1;\n\t"
        "mov.b64            t, {m, 0};\n\t"
        "sub.u64            %0, %0, t;\n\t"
        "}"
        : "+l"(r)
        : "l"(a.val_), "l"(b.val_));
    val_ = r;
  }

  /** Multiplication in FF(P): val_ = a * b mod P. */
  __device__ inline
  void Mul(const FFP& a, const FFP& b) {
    asm("{\n\t"
        ".reg .u32          r0, r1;\n\t"
        ".reg .u32          m0, m1, m2, m3;\n\t"
        ".reg .u64          t;\n\t"
        ".reg .pred         p, q;\n\t"
        // 128-bit = 64-bit * 64-bit
        "mul.lo.u64         t, %1, %2;\n\t"
        "mov.b64            {m0, m1}, t;\n\t"
        "mul.hi.u64         t, %1, %2;\n\t"
        "mov.b64            {m2, m3}, t;\n\t"
        // 128-bit mod P with add / sub
        "add.u32            r1, m1, m2;\n\t"
        "sub.cc.u32         r0, m0, m2;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "sub.cc.u32         r0, r0, m3;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        // fix result
        "setp.eq.u32        p|q, m2, 0;\n\t"
        "mov.b64            t, {m0, m1};\n\t"
        // ret -= (uint32_t)(-(ret > mul[0] && m[2] == 0));
        "set.gt.and.u32.u64 m3, %0, t, p;\n\t"
        "sub.cc.u32         r0, r0, m3;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        // ret += (uint32_t)(-(ret < mul[0] && m[2] != 0));
        "set.lt.and.u32.u64 m3, %0, t, q;\n\t"
        "add.cc.u32         r0, r0, m3;\n\t"
        "addc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        "}"
        : "+l"(val_)
        : "l"(a.val_), "l"(b.val_));
    ModP(val_);
  }

  /** \brief Exponentiation in FF(P): val_ = val_ ^ e mod P. */
  __device__ inline
  void Pow(uint32_t e) {
    if (0 == e) {
      val_ = 1;
      return;
    }
    FFP y = 1;
    uint64_t n = (uint64_t)e;
    while (n > 1) {
      if (0 != (n & 0x1))
        y *= (*this);
      *this *= (*this);
      n >>= 1;
    }
    *this *= y;
  }

  /** \brief Exponentiation in FF(P): return a ^ e mod P. */
  __device__ inline
  static FFP Pow(const FFP& a, uint32_t e) {
    FFP r = a;
    r.Pow(e);
    return r;
  }

  /**
   * Binary left shifting in FF(P): val_ = val_ * 2^l mod P.
   * @param[in] l An integer in [0, 32)
   */
  __device__ inline
  void Lsh32(uint32_t l) {
    asm("{\n\t"
        ".reg .u32      r0, r1;\n\t"
        ".reg .u32      t0, t1, t2;\n\t"
        ".reg .u32      n;\n\t"
        ".reg .u64      s;\n\t"
        // t[2] = (uint32_t)(x >> (64-l));
        // t[1] = (uint32_t)(x >> (32-l));
        // t[0] = (uint32_t)(x << l);
        "mov.b64        {r0, r1}, %0;\n\t"
        "shl.b32        t0, r0, %1;\n\t"
        "sub.u32        n, 32, %1;\n\t"
        "shr.b64        s, %0, n;\n\t"
        "mov.b64        {t1, t2}, s;\n\t"
        // mod P
        "add.u32        r1, t1, t2;\n\t"
        "sub.cc.u32     r0, t0, t2;\n\t"
        "subc.u32       r1, r1, 0;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        // ret += (uint32_t)(-(ret < ((uint64_t *)t)[0]));
        "mov.b64        s, {t0, t1};\n\t"
        "set.lt.u32.u64 t2, %0, s;\n\t"
        "add.cc.u32     r0, r0, t2;\n\t"
        "addc.u32       r1, r1, 0;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        "}"
        : "+l"(val_)
        : "r"(l));
    // ret += (uint32_t)(-(ret >= FFP_MODULUS));
    ModP(val_);
  }

  /**
   * Binary left shifting in FF(P): val_ = val_ * 2^l mod P.
   * @param[in] l An integer in [32, 64)
   */
  __device__ inline
  void Lsh64(uint32_t l) {
    asm("{\n\t"
        ".reg .u32          r0, r1;\n\t"
        ".reg .u32          t0, t1, t2;\n\t"
        ".reg .u32          n;\n\t"
        ".reg .u64          s;\n\t"
        ".reg .pred         p, q;\n\t"
        // t[2] = (uint32_t)(x >> (96-l));
        // t[1] = (uint32_t)(x >> (64-l));
        // t[0] = (uint32_t)(x << (l-32));
        "mov.b64            {r0, r1}, %0;\n\t"
        "sub.u32            n, %1, 32;\n\t"
        "shl.b32            t0, r0, n;\n\t"
        "sub.u32            n, 32, n;\n\t"
        "shr.b64            s, %0, n;\n\t"
        "mov.b64            {t1, t2}, s;\n\t"
        // mod P
        "add.u32            r1, t0, t1;\n\t"
        "sub.cc.u32         r0, 0, t1;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "sub.cc.u32         r0, r0, t2;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        // ret -= (uint32_t)(-(ret > ((uint64_t)t[0] << 32) && t[1] == 0));
        "setp.eq.u32        p|q, t1, 0;\n\t"
        "mov.b64            s, {0, t0};\n\t"
        "set.gt.and.u32.u64 t2, %0, s, p;\n\t"
        "sub.cc.u32         r0, r0, t2;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        // ret += (uint32_t)(-(ret < ((uint64_t)t[0] << 32) && t[1] != 0));
        "set.lt.and.u32.u64 t2, %0, s, q;\n\t"
        "add.cc.u32         r0, r0, t2;\n\t"
        "addc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        "}"
        : "+l"(val_)
        : "r"(l));
    // ret += (uint32_t)(-(ret >= FFP_MODULUS));
    ModP(val_);
  }

  /**
   * Binary left shifting in FF(P): val_ = val_ * 2^l mod P.
   * @param[in] l An integer in [64, 96)
   */
  __device__ inline
  void Lsh96(uint32_t l) {
    asm("{\n\t"
        ".reg .u32      r0, r1;\n\t"
        ".reg .u32      t0, t1, t2;\n\t"
        ".reg .u32      n;\n\t"
        ".reg .u64      s;\n\t"
        // t[2] = (uint32_t)(x >> (128-l));
        // t[1] = (uint32_t)(x >> (96-l));
        // t[0] = (uint32_t)(x << (l-64));
        "mov.b64        {r0, r1}, %0;\n\t"
        "sub.u32        n, %1, 64;\n\t"
        "shl.b32        t0, r0, n;\n\t"
        "sub.u32        n, 32, n;\n\t"
        "shr.b64        s, %0, n;\n\t"
        "mov.b64        {t1, t2}, s;\n\t"
        // mod P
        "add.cc.u32     r0, t1, t0;\n\t"
        "addc.u32       r1, t2, 0;\n\t"
        "sub.u32        r1, r1, t0;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        // ret -= (uint32_t)(-(ret > ((uint64_t *)t)[1]));
        "mov.b64        s, {t1, t2};\n\t"
        "set.gt.u32.u64 t2, %0, s;\n\t"
        "sub.cc.u32     r0, r0, t2;\n\t"
        "subc.u32       r1, r1, 0;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        "}"
        : "+l"(val_)
        : "r"(l));
    // ret += (uint32_t)(-(ret >= FFP_MODULUS));
    ModP(val_);
    val_ = kModulus_ - val_;
  }

  /**
   * Binary left shifting in FF(P): val_ = val_ * 2^l mod P.
   * @param[in] l An integer in [96, 128)
   */
  __device__ inline
  void Lsh128(uint32_t l) {
    asm("{\n\t"
        ".reg .u32      r0, r1;\n\t"
        ".reg .u32      t0, t1, t2;\n\t"
        ".reg .u32      n;\n\t"
        ".reg .u64      s;\n\t"
        // t[2] = (uint32_t)(x >> (160-l));
        // t[1] = (uint32_t)(x >> (128-l));
        // t[0] = (uint32_t)(x << (l-96));
        "mov.b64        {r0, r1}, %0;\n\t"
        "sub.u32        n, %1, 96;\n\t"
        "shl.b32        t0, r0, n;\n\t"
        "sub.u32        n, 32, n;\n\t"
        "shr.b64        s, %0, n;\n\t"
        "mov.b64        {t1, t2}, s;\n\t"
        // mod P
        "add.u32        r1, t1, t2;\n\t"
        "sub.cc.u32     r0, t0, t2;\n\t"
        "subc.u32       r1, r1, 0;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        // ret += (uint32_t)(-(ret < ((uint64_t *)t)[0]));
        "mov.b64        s, {t0, t1};\n\t"
        "set.lt.u32.u64 t2, %0, s;\n\t"
        "add.cc.u32     r0, r0, t2;\n\t"
        "addc.u32       r1, r1, 0;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        "}"
        : "+l"(val_)
        : "r"(l));
    // ret += (uint32_t)(-(ret >= FFP_MODULUS));
    ModP(val_);
    val_ = kModulus_ - val_;
  }

  /**
   * Binary left shifting in FF(P): val_ = val_ * 2^l mod P.
   * @param[in] l An integer in [128, 160)
   */
  __device__ inline
  void Lsh160(uint32_t l) {
    asm("{\n\t"
        ".reg .u32          r0, r1;\n\t"
        ".reg .u32          t0, t1, t2;\n\t"
        ".reg .u32          n;\n\t"
        ".reg .u64          s;\n\t"
        ".reg .pred         p, q;\n\t"
        // t[2] = (uint32_t)(x >> (192-l));
        // t[1] = (uint32_t)(x >> (160-l));
        // t[0] = (uint32_t)(x << (l-128));
        "mov.b64            {r0, r1}, %0;\n\t"
        "sub.u32            n, %1, 128;\n\t"
        "shl.b32            t0, r0, n;\n\t"
        "sub.u32            n, 32, n;\n\t"
        "shr.b64            s, %0, n;\n\t"
        "mov.b64            {t1, t2}, s;\n\t"
        // mod P
        "add.u32            r1, t0, t1;\n\t"
        "sub.cc.u32         r0, 0, t1;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "sub.cc.u32         r0, r0, t2;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        // ret -= (uint32_t)(-(ret > ((uint64_t)t[0] << 32) && t[1] == 0));
        "setp.eq.u32        p|q, t1, 0;\n\t"
        "mov.b64            s, {0, t0};\n\t"
        "set.gt.and.u32.u64 t2, %0, s, p;\n\t"
        "sub.cc.u32         r0, r0, t2;\n\t"
        "subc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        // ret += (uint32_t)(-(ret < ((uint64_t)t[0] << 32) && t[1] != 0));
        "set.lt.and.u32.u64 t2, %0, s, q;\n\t"
        "add.cc.u32         r0, r0, t2;\n\t"
        "addc.u32           r1, r1, 0;\n\t"
        "mov.b64            %0, {r0, r1};\n\t"
        "}"
        : "+l"(val_)
        : "r"(l));
    // ret += (uint32_t)(-(ret >= FFP_MODULUS));
    ModP(val_);
    val_ = kModulus_ - val_;
  }

  /**
   * Binary left shifting in FF(P): val_ = val_ * 2^l mod P.
   * @param[in] l An integer in [160, 192)
   */
  __device__ inline
  void Lsh192(uint32_t l) {
    asm("{\n\t"
        ".reg .u32      r0, r1;\n\t"
        ".reg .u32      t0, t1, t2;\n\t"
        ".reg .u32      n;\n\t"
        ".reg .u64      s;\n\t"
        // t[2] = (uint32_t)(x << (l-160));
        // t[1] = (uint32_t)(x >> (224-l));
        // t[0] = (uint32_t)(x >> (192-l));
        "mov.b64        {r0, r1}, %0;\n\t"
        "sub.u32        n, %1, 160;\n\t"
        "shl.b32        t2, r0, n;\n\t"
        "sub.u32        n, 32, n;\n\t"
        "shr.b64        s, %0, n;\n\t"
        "mov.b64        {t0, t1}, s;\n\t"
        // mod P
        "add.cc.u32     r0, t0, t2;\n\t"
        "addc.u32       r1, t1, 0;\n\t"
        "sub.u32        r1, r1, t2;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        // ret += (uint32_t)(-(ret > ((uint64_t *)t)[0]));
        "mov.b64        s, {t0, t1};\n\t"
        "set.gt.u32.u64 t2, %0, s;\n\t"
        "sub.cc.u32     r0, r0, t2;\n\t"
        "subc.u32       r1, r1, 0;\n\t"
        "mov.b64        %0, {r0, r1};\n\t"
        "}"
        : "+l"(val_)
        : "r"(l));
    // ret += (uint32_t)(-(ret >= FFP_MODULUS));
    ModP(val_);
  }

}; // class FFP

} // namespace cufhe
