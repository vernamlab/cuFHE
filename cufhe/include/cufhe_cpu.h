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

#include "cufhe.h"

namespace cufhe {

void And (Ctxt& out, const Ctxt& in0, const Ctxt& in1, const PubKey& pub_key);
void Or  (Ctxt& out, const Ctxt& in0, const Ctxt& in1, const PubKey& pub_key);
void Nand(Ctxt& out, const Ctxt& in0, const Ctxt& in1, const PubKey& pub_key);
void Nor (Ctxt& out, const Ctxt& in0, const Ctxt& in1, const PubKey& pub_key);
void Xor (Ctxt& out, const Ctxt& in0, const Ctxt& in1, const PubKey& pub_key);
void Xnor(Ctxt& out, const Ctxt& in0, const Ctxt& in1, const PubKey& pub_key);
void Not (Ctxt& out, const Ctxt& in);
void Copy(Ctxt& out, const Ctxt& in);
// Not Ready...
// void Mux(Ctxt& out, const Ctxt& in0, const Ctxt& in1, const Ctxt& in2,
//          cudaStream_t st = 0);

} // namespace cufhe
