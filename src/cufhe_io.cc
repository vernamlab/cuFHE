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

#include <fstream>

#include "../include/cufhe.h"
#include "../include/cufhe_core.h"

namespace cufhe {

template <typename T>
void WriteStream(std::ofstream& out, const DataTemplate<T>& data)
{
    if (out.is_open())
        for (int i = 0; i < data.SizeData() / sizeof(T); i++)
            out << std::to_string(data.data()[i]) << "\n";
}

template <typename T>
void ReadStream(const DataTemplate<T>& data, std::ifstream& in)
{
    std::string value;
    if (in.is_open())
        for (int i = 0; i < data.SizeData() / sizeof(T); i++)
            if (std::getline(in, value)) data.data()[i] = stoi(value);
}

void WritePriKeyToFile(const PriKey& pri_key, FileName file)
{
    std::ofstream stream(file);
    WriteStream<Binary>(stream, *pri_key.lwe_key_);
    WriteStream<Binary>(stream, *pri_key.tlwe_key_);
    stream.close();
}

void ReadPriKeyFromFile(PriKey& pri_key, FileName file)
{
    std::ifstream stream(file);
    ReadStream<Binary>(*pri_key.lwe_key_, stream);
    ReadStream<Binary>(*pri_key.tlwe_key_, stream);
    stream.close();
}

void WritePubKeyToFile(const PubKey& pub_key, FileName file)
{
    std::ofstream stream(file);
    WriteStream<Torus>(stream, *pub_key.bk_);
    for (int i = 0; i < pub_key.ksk_->NumLWESamples(); i++)
        WriteStream<Torus>(stream, pub_key.ksk_->ExtractLWESample(i));
    stream.close();
}

void ReadPubKeyFromFile(PubKey& pub_key, FileName file)
{
    std::ifstream stream(file);
    ReadStream<Torus>(*pub_key.bk_, stream);
    for (int i = 0; i < pub_key.ksk_->NumLWESamples(); i++)
        ReadStream<Torus>(pub_key.ksk_->ExtractLWESample(i), stream);
    stream.close();
}

void WriteCtxtToFile(const Ctxt& ct, FileName file)
{
    std::ofstream stream(file);
    WriteStream<Torus>(stream, *ct.lwe_sample_);
    stream.close();
}

void ReadCtxtFromFile(Ctxt& ct, FileName file)
{
    std::ifstream stream(file);
    ReadStream<Torus>(*ct.lwe_sample_, stream);
    stream.close();
}

}  // namespace cufhe
