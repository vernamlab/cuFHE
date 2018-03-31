/**
 * Copyright 2018 Gizem S. Cetin <gscetin@wpi.edu>
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

#include "cufhe.h"
#include "cufhe_gpu.cuh"
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/class.hpp>

using namespace cufhe;


BOOST_PYTHON_MODULE(pyfhe_gpu)
{
        using namespace boost::python;

        /* DATA/STRUCTS */

	class_<Param>("Param")
		.def(init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, double, double>())
		.def_readwrite("lwe_n", &Param::lwe_n_)
		.def_readwrite("tlwe_n", &Param::tlwe_n_)
		.def_readwrite("tlwe_k", &Param::tlwe_k_)
		.def_readwrite("tgsw_decomp_bits", &Param::tgsw_decomp_bits_)
		.def_readwrite("tgsw_decomp_size", &Param::tgsw_decomp_size_)
		.def_readwrite("keyswitching_decomp_bits", &Param::keyswitching_decomp_bits_)
		.def_readwrite("keyswitching_decomp_size", &Param::keyswitching_decomp_size_)
		.def_readwrite("lwe_noise", &Param::lwe_noise_)
		.def_readwrite("tlwe_noise", &Param::tlwe_noise_);


        object Ptxt_class = class_<Ptxt>("Ptxt")
		.add_property("message", &Ptxt::get, &Ptxt::set);
                Ptxt_class.attr("PtxtSpace") = Ptxt::kPtxtSpace;

        class_<Ctxt>("Ctxt", init< optional<bool> >());
	class_<PubKey>("PubKey", init< optional<bool> >());
	class_<PriKey>("PriKey", init< optional<bool> >());

	class_<Stream>("Stream")
		.def(init<int>())
		.def("Create", &Stream::Create)
		.def("Destroy", &Stream::Destroy);

        /* SERVER METHODS */
	def("Initialize", Initialize);
	def("CleanUp", CleanUp);
	def("Synchronize", Synchronize);

	/* GPU GATES */
	def("AND", And);
	def("OR", Or);
	def("NAND", Nand);
	def("NOR", Nor);
	def("XOR", Xor);
	def("XNOR", Xnor);
	def("NOT", Not);
	def("Copy", Copy);


        /* CLIENT METHODS */

        def("SetSeed", SetSeed);
        def("PriKeyGen", PriKeyGen);
        def("PubKeyGen", PubKeyGen);
        def("KeyGen", KeyGen);
        def("Encrypt", Encrypt);
        def("Decrypt", Decrypt);


        /* I/O METHODS */

        def("WritePriKeyToFile", WritePriKeyToFile);
        def("ReadPriKeyFromFile", ReadPriKeyFromFile);
        def("WritePubKeyToFile", WritePubKeyToFile);
        def("ReadPubKeyFromFile", ReadPubKeyFromFile);
        def("WriteCtxtToFile", WriteCtxtToFile);
        def("ReadCtxtFromFile", ReadCtxtFromFile);

}
