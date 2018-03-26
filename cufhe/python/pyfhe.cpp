#include "cufhe.h"
#include "cufhe_cpu.h"
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/class.hpp>

using namespace cufhe;


BOOST_PYTHON_MODULE(pyfhe)
{
        using namespace boost::python;

        /* DATA/STRUCTS */

	class_<Param>("Param")
		.def(init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, double, double>())
		.def_readwrite("lwe_n_", &Param::lwe_n_)
		.def_readwrite("tlwe_n_", &Param::tlwe_n_)
		.def_readwrite("tlwe_k_", &Param::tlwe_k_)
		.def_readwrite("tgsw_decomp_bits_", &Param::tgsw_decomp_bits_)
		.def_readwrite("tgsw_decomp_size_", &Param::tgsw_decomp_size_)
		.def_readwrite("keyswitching_decomp_bits_", &Param::keyswitching_decomp_bits_)
		.def_readwrite("keyswitching_decomp_size_", &Param::keyswitching_decomp_size_)
		.def_readwrite("lwe_noise_", &Param::lwe_noise_)
		.def_readwrite("tlwe_noise_", &Param::tlwe_noise_);

	//def("GetDefaultParam", GetDefaultParam, return_value_policy<manage_new_object>()); // Returning a pointer gives "free() invalid pointer" error in python

        object Ptxt_class = class_<Ptxt>("Ptxt")
		.add_property("message_", &Ptxt::get, &Ptxt::set);
                Ptxt_class.attr("PtxtSpace") = Ptxt::kPtxtSpace;

        class_<Ctxt>("Ctxt", init< optional<bool> >());
	class_<PubKey>("PubKey", init< optional<bool> >());
	class_<PriKey>("PriKey", init< optional<bool> >());


        /* SERVER METHODS */
	/* CPU GATES */
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

