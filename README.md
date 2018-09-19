# cuFHE
CUDA-accelerated Fully Homomorphic Encryption Library

v1.0_beta -- release on Mar/14/2018

## What is cuFHE?
The cuFHE library is an open-source library for Fully Homomorphic Encryption (FHE) on CUDA-enabled GPUs. It implements the TFHE scheme [CGGI16][CGGI17] proposed by Chillotti et al. in CUDA C++. Compared to the [TFHE lib](https://github.com/tfhe/tfhe) which reports the fastest gate-by-gate bootstrapping performance on CPUs, the cuFHE library yields roughly 20 times of speedup on an NVIDIA Titan Xp graphics card. The cuFHE library benefits greatly from an improved CUDA implementation of the number-theoretic transform (NTT) proposed in the [cuHE library](https://github.com/vernamlab/cuHE) [Dai15] by Dai and Sunar.

| [TFHE lib](https://github.com/tfhe/tfhe) | cuFHE | Speedup |
|---|---|---|
| 13 ms | **0.5 ms** | 26 times |

### System Requirements
**The library has been tested on Ubuntu Desktop 16.04 only.**
This "Makefile" is created for Linux systems. Please create your own Makefile for MacOS and Windows. We are working on cross-platform support.

GPU support requires NVIDIA Driver, NVIDIA CUDA Toolkit and a GPU with **Compute Capability no less than 6.0**.
For devices with Compute Capability less than 6.0, there is [an issue](https://github.com/vernamlab/cuFHE/issues/2) that have not been solved yet. Any fix or suggestion is welcomed.

### Installation (Linux)
- Run `make` from the directory `cufhe/` for default compilation. This will
  1. create directories `build` and `bin`,
  2. generate shared libraries `libcufhe_cpu.so` (CPU standalone),
  3. `libcufhe_gpu.so` (GPU support) in `bin` directory, and 3) create test and benchmarking executables `test_api_cpu` and `test_api_gpu` in `bin`.

- Alternatively, run `make cpu` or `make gpu` for individual library and executable.
- Copy the library files and `include` folder to any desirable location. Remember to export your library directory with `export LD_LIBRARY_PATH=directory`. Run `test_api_gpu` to see the latency per gate.
- We provide a Python wrapper which uses boost-python tool. To use the Python interface, you will need
  1. a python interpreter, (probably in `/usr/bin/`)
  2. boost-python library, (Run `sudo apt-get install libboost-python-dev`, if you don't have it installed.)
  3. to change the Makefile if your python and boost include/lib paths are different than default,
  4. to run `make python_cpu` for CPU library and `make python_gpu` for GPU library, and finally
  5. to test the python scripts under `cufhe/python/`.

### User Manual
Use files in `cufhe/test/` as examples. To summarize, follow the following function calling procedures.
```c++
SetSeed(); // init random generator seed
PriKey pri_key;
PubKey pub_key;
KeyGen(pub_key, pri_key); // key generation
// alternatively, write / read key files
Ptxt pt[2];
pt[0] = 0; // 0 or 1, single bit
pt[1] = 1;
Ctxt ct[2];
Encrypt(ct[0], pt[0], pri_key);
Encrypt(ct[1], pt[1], pri_key);

Initialize(pub_key); // for GPU library
Nand(ct[0], ct[0], ct[1], pub_key); // for CPU library
Nand(ct[0], ct[0], ct[1]); // for GPU library non-parallelized gates
cudaSteam_t stream_id;
cudaStreamCreate(&stream_id);
Nand(ct[0], ct[0], ct[1], stream_id); // for GPU library parallelized gates

Decrypt(pt[0], ct[0], pri_key);
CleanUp(); // for GPU library
```

Currently implemented gates are `And, Or, Nand, Nor, Xor, Xnor, Not, Copy`.

## Change Log
- **version 1.0_beta** -- released on Mar/14/2018.
  - Supports single-bit unpacked encryption / decryption / gates.
  - C++ interface with CPU and GPU separate libraries.

## Acknowledgement
- We appreciate any bug reports or compiling issues.
- Dai and Sunar’s work was in part provided by the US National Science Foundation CNS Award #1319130 and #1561536.
- We gratefully acknowledge the support of NVIDIA Corporation with the donation of the Titan X Pascal GPU used for this research.

## Reference
[CGGI16]: Chillotti, I., Gama, N., Georgieva, M., & Izabachene, M. (2016, December). Faster fully homomorphic encryption: Bootstrapping in less than 0.1 seconds. In International Conference on the Theory and Application of Cryptology and Information Security (pp. 3-33). Springer, Berlin, Heidelberg.

[CGGI17]: Chillotti, I., Gama, N., Georgieva, M., & Izabachène, M. (2017, December). Faster Packed Homomorphic Operations and Efficient Circuit Bootstrapping for TFHE. In International Conference on the Theory and Application of Cryptology and Information Security (pp. 377-408). Springer, Cham.

[Dai15]: Dai, W., & Sunar, B. (2015, September). cuHE: A homomorphic encryption accelerator library. In International Conference on Cryptography and Information Security in the Balkans (pp. 169-186). Springer, Cham.
